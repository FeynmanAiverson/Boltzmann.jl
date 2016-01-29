using Distributions
using ProgressMeter
using Base.LinAlg.BLAS
using Compat
using Devectorize
using HDF5
using PyCall
@pyimport matplotlib.pyplot as plt
@pyimport numpy as np

import StatsBase.fit

function get_negative_samples(dbm::DBM, vis_init::Mat{Float64}, array_hid_init::Array{Array{Float64},1}, approx::AbstractString, iterations::Int)
    if approx=="naive" || contains(approx,"tap")
        v_neg, array_h_neg = equilibrate(dbm,vis_init,array_hid_init; iterations=iterations, approx=approx)
    end
    if approx=="CD" || approx=="mixed"
        # In the case of Gibbs/MCMC sampling, we will take the binary visible samples as the negative
        # visible samples, and the expectation (means) for the negative hidden samples.
        v_neg, array_h_neg, _, _=MCMC(dbm, vis_init, array_hid_init; iterations=iterations, StartMode="visible")
    end

    return v_neg, array_h_neg
end

#### Unlike for RBMs, the computations of positive samples is not straightforward for DBMs and requires an equilibration. This function only exists for DBMs.
function get_positive_samples(dbm::DBM, vis::Mat{Float64}, array_hid_init::Array{Array{Float64},1},approx::AbstractString, iterations::Int)
    if approx=="naive" || contains(approx,"tap")
        v_pos, array_h_pos = clamped_equilibrate(dbm,vis,array_hid_init; iterations=iterations, approx=approx)
    end

    if approx=="mixed"
        v_pos, array_h_pos = clamped_equilibrate(dbm,vis,array_hid_init; iterations=iterations, approx="naive")
    end

    if approx=="CD" 
        # In the case of Gibbs/MCMC sampling, we will take the binary visible samples as the negative
        # visible samples, and the expectation (means) for the negative hidden samples.
        v_pos, array_h_pos, _, _=MCMC_clamped(dbm, vis, array_hid_init; iterations=iterations, StartMode="visible") 
    end

    return v_pos, array_h_pos
end

function fit_batch!(dbm::DBM, vis::Mat{Float64};
                    persistent=true, lr=0.1, NormalizationApproxIter=1,
                    weight_decay="none",decay_magnitude=0.01, approx="CD")
    depth = length(dbm)

    array_h_pos_init = ProbHidInitCondOnVis(dbm, vis)
    v_pos, array_h_pos = get_positive_samples(dbm, vis, array_h_pos_init, approx, NormalizationApproxIter)

    if persistent # Set starting points in the case of persistence
        v_init = copy(dbm[1].persistent_chain_vis)  
        array_hid_init = Array(Array{Float64}, depth) 
        for l=1:depth  
            array_hid_init[l] = copy(dbm[l].persistent_chain_hid)
        end
    else
        v_init = vis
        array_hid_init = array_h_pos
    end    

    # Calculate the negative samples according to the desired approximation mode
    v_neg, array_h_neg = get_negative_samples(dbm,v_init,array_hid_init,approx,NormalizationApproxIter)

    # If we are in persistent mode, update the chain accordingly
    if persistent
        copy!(dbm[1].persistent_chain_vis,v_neg)
        copy!(dbm[1].persistent_chain_hid,array_h_neg[1])
        for l=2:depth  
              copy!(dbm[l].persistent_chain_vis,array_h_neg[l-1])
              copy!(dbm[l].persistent_chain_hid,array_h_neg[l])
        end
    end

    # Update on weights and biases
    # Start with the first RBM
    rbm=dbm[1]
    h_pos=array_h_pos[1]
    h_neg=array_h_neg[1]
    calculate_weight_gradient!(rbm,h_pos,v_pos,h_neg,v_neg,lr,approx=approx)
    if weight_decay == "l2"
        regularize_weight_gradient!(rbm,lr;L2Penalty=decay_magnitude)
    end
    if weight_decay == "l1"
        regularize_weight_gradient!(rbm,lr;L1Penalty=decay_magnitude)
    end
    update_weights!(rbm,approx)
    rbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    rbm.vbias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))

    # Bottom-up pass of updates
    for l=2:depth 
        rbm=dbm[l]
        v_pos,h_pos=array_h_pos[l-1:l]
        v_neg,h_neg=array_h_neg[l-1:l]
        calculate_weight_gradient!(rbm,h_pos,v_pos,h_neg,v_neg,lr,approx=approx)
        if weight_decay == "l2"
            regularize_weight_gradient!(rbm,lr;L2Penalty=decay_magnitude)
        end
        if weight_decay == "l1"
            regularize_weight_gradient!(rbm,lr;L1Penalty=decay_magnitude)
        end
        update_weights!(rbm,approx)
        rbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
        rbm.vbias = copy(dbm[l-1].hbias) ## Constraining biases of visible units to be equal to biases of hidden units from shallower layer
    end
    return dbm
end


"""
    # Boltzmann.fit (TrainingDBM.jl)
    ## Function Call
        `fit(dbm::DBM, X::Mat{Float64}[, persistent, lr, batch_size, NormalizationApproxIter, weight_decay, 
                                         decay_magnitude, validation,monitor_ever, monitor_vis,
                                         approx, persistent_start])`
    ## Description
    The core DBM training function. Learns the weights and biasings using 
    either standard Contrastive Divergence (CD) or Persistent CD, depending on
    the user options. 
    
    - *dbm:* DBM object
    - *X:* Set of training vectors

    ### Optional Inputs
     - *persistent:* Whether or not to use persistent-CD [default=true]
     - *persistent_start:* At which epoch to start using the persistent chains. Only
                           applicable for the case that `persistent=true`.
                           [default=1]
     - *lr:* Learning rate [default=0.1]
     - *n_iter:* Number of training epochs [default=10]
     - *batch_size:* Minibatch size [default=100]
     - *NormalizationApproxIter:* Number of Gibbs sampling steps on the Markov Chain [default=1]
     - *weight_decay:* A string value representing the regularization to add to apply to the 
                       weight magnitude during training {"none","l1","l2"}. [default="none"]
     - *decay_magnitude:* Relative importance assigned to the weight regularization. Smaller
                          values represent less regularization. Should be in range (0,1). 
                          [default=0.01]
     - *validation:* An array of validation samples, e.g. a held out set of training data.
                     If passed, `fit` will also track generalization progress during training.
                     [default=empty-set]
     - *score_every:* Controls at which epoch the progress of the fit is monitored. Useful to 
                      speed up the fit procedure if detailed progress monitoring is not required.
                      [default=5]

    ## Returns
     - *::DBM* -- A trained DBM model.

"""
     # - *::Monitor* -- A Monitor structure containing information on the training progress over
     #                  epochs.

function fit(dbm::DBM, X::Mat{Float64};
             persistent=true, lr=0.1, n_iter=10, batch_size=100, NormalizationApproxIter=1,
             weight_decay="none",decay_magnitude=0.01,validation=[],
             monitor_every=5,monitor_vis=false, approx="CD",
             persistent_start=1)

    # TODO: This line needs to be changed to accomodate real-valued units
    @assert minimum(X) >= 0 && maximum(X) <= 1
    depth=length(dbm)
    n_valid=0
    n_features = size(X, 1)
    n_samples = size(X, 2)
    n_hidden = size(dbm[1].W,1)
    for l=2:depth 
        n_hidden+=size(dbm[l].W,1)
    end
    n_batches = @compat Int(ceil(n_samples / batch_size))
    N = n_hidden+n_features

    # Check for the existence of a validation set
    flag_use_validation=false
    if length(validation)!=0
        flag_use_validation=true
        n_valid=size(validation,2)        
    end

    # Create the historical monitor
    ProgressMonitor = Monitor(n_iter,monitor_every;monitor_vis=monitor_vis,
                                                   validation=flag_use_validation)

    # # Print info to user
    m_ = dbm[1].momentum
    info("=====================================")
    info("DBM Training")
    info("=====================================")
    info("  + Training Samples:     $n_samples")
    info("  + Features:             $n_features")
    info("  + Hidden Units:         $n_hidden")
    info("  + Hidden Layers:        $depth")
    info("  + Epochs to run:        $n_iter")
    info("  + Persistent ?:         $persistent")
    info("  + Training approx:      $approx")
    info("  + Momentum:             $m_")
    info("  + Learning rate:        $lr")
    info("  + Norm. Approx. Iters:  $NormalizationApproxIter")   
    info("  + Weight Decay?:        $weight_decay") 
    info("  + Weight Decay Mag.:    $decay_magnitude")
    info("  + Validation Set?:      $flag_use_validation")    
    info("  + Validation Samples:   $n_valid")   
    info("=====================================")

    # Scale the learning rate by the batch size
    lr=lr/batch_size

    # Random initialization of the persistent chains
    dbm[1].persistent_chain_vis,_ = random_columns(X,batch_size)
    array_persistent_chain_hid = ProbHidInitCondOnVis(dbm,dbm[1].persistent_chain_vis)
    dbm[1].persistent_chain_hid = copy(array_persistent_chain_hid[1])
    for l=2:depth
        dbm[l].persistent_chain_vis = copy(array_persistent_chain_hid[l-1])  
        dbm[l].persistent_chain_hid = copy(array_persistent_chain_hid[l])
    end

    use_persistent = false
    for itr=1:n_iter
        # Check to see if we can use persistence at this epoch
        use_persistent = itr>=persistent_start ? persistent : false

        tic()

        # Mini-batch fitting loop. 
        @showprogress 1 "Fitting Batches..." for i=1:n_batches
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            batch = full(batch)
          
            fit_batch!(dbm, batch; persistent=use_persistent, 
                                   NormalizationApproxIter=NormalizationApproxIter,
                                   weight_decay=weight_decay,
                                   decay_magnitude=decay_magnitude,
                                   lr=lr, approx=approx)
            
        end
        
        # Get the average wall-time in µs
        walltime_µs=(toq()/n_batches/N)*1e6
        
        UpdateMonitor!(dbm,ProgressMonitor,X,itr;bt=walltime_µs,validation=validation)
        ShowMonitor(dbm,ProgressMonitor,X,itr)
    end

    return dbm, ProgressMonitor
end

"""
    # Boltzmann.fit_doubled (TrainingDBM.jl)
    ## Function Call
        `fit_doubled(dbm::DBM, X::Mat{Float64}, persistent, lr, batch_size, NormalizationApproxIter, weight_decay, 
                                         decay_magnitude, validation,monitor_ever, monitor_vis,
                                         approx, persistent_start])`
    ## Description
    The core RBM training function. Learns the weights and biasings using 
    either standard Contrastive Divergence (CD) or Persistent CD, depending on
    the user options. 
    
    - *rbm:* RBM object
    - *X:* Set of training vectors
    - *which*

    ### Optional Inputs
     - *persistent:* Whether or not to use persistent-CD [default=true]
     - *persistent_start:* At which epoch to start using the persistent chains. Only
                           applicable for the case that `persistent=true`.
                           [default=1]
     - *lr:* Learning rate [default=0.1]
     - *n_iter:* Number of training epochs [default=10]
     - *batch_size:* Minibatch size [default=100]
     - *NormalizationApproxIter:* Number of Gibbs sampling steps on the Markov Chain [default=1]
     - *weight_decay:* A string value representing the regularization to add to apply to the 
                       weight magnitude during training {"none","l1","l2"}. [default="none"]
     - *decay_magnitude:* Relative importance assigned to the weight regularization. Smaller
                          values represent less regularization. Should be in range (0,1). 
                          [default=0.01]
     - *validation:* An array of validation samples, e.g. a held out set of training data.
                     If passed, `fit` will also track generalization progress during training.
                     [default=empty-set]
     - *score_every:* Controls at which epoch the progress of the fit is monitored. Useful to 
                      speed up the fit procedure if detailed progress monitoring is not required.
                      [default=5]

    ## Returns
     - *::RBM* -- A trained RBM model.
"""

function fit_doubled(rbm,X::Mat{Float64}, which::AbstractString;
             persistent=true, lr=0.1, n_iter=10, batch_size=100, NormalizationApproxIter=1,
             weight_decay="none",decay_magnitude=0.01,validation=[],
             monitor_every=5,monitor_vis=false, approx="CD",
             persistent_start=1)

    # TODO: This line needs to be changed to accomodate real-valued units
    @assert minimum(X) >= 0 && maximum(X) <= 1

    n_valid=0
    n_features = size(X, 1)
    n_samples = size(X, 2)
    n_hidden = size(rbm.W,1)
    n_batches = @compat Int(ceil(n_samples / batch_size))
    N = n_hidden+n_features
    
    if which=="input"
        rbmaux = BernoulliRBM(2*n_features, n_hidden, (2*rbm.VisShape[1],rbm.VisShape[2]); momentum=rbm.momentum, sigma = 0.01)
        rbmaux.W = [rbm.W rbm.W]
        X = [X ; X]

    elseif  which=="output"
        rbmaux = BernoulliRBM(n_features, 2*n_hidden, (n_features,1); momentum=rbm.momentum, sigma = 0.01)
        rbmaux.W = [rbm.W ; rbm.W]
    end

    # Check for the existence of a validation set
    flag_use_validation=false
    # if length(validation)!=0
    #     flag_use_validation=true
    #     n_valid=size(validation,2)        
    # end

    # Create the historical monitor
    ProgressMonitor = Monitor(n_iter,monitor_every;monitor_vis=monitor_vis,
                                                   validation=flag_use_validation)
    # ProgressMonitor.FigureHandle=plt.figure("DBM";figsize=(12,15))

    # Print info to user
    m_ = rbmaux.momentum
    info("=====================================")
    info("RBM Training")
    info("=====================================")
    info("  + Training Samples:     $n_samples")
    info("  + Features:             $n_features")
    info("  + Hidden Units:         $n_hidden")
    info("  + Epochs to run:        $n_iter")
    info("  + Persistent ?:         $persistent")
    info("  + Training approx:      $approx")
    info("  + Momentum:             $m_")
    info("  + Learning rate:        $lr")
    info("  + Norm. Approx. Iters:  $NormalizationApproxIter")   
    info("  + Weight Decay?:        $weight_decay") 
    info("  + Weight Decay Mag.:    $decay_magnitude")
    info("  + Validation Set?:      $flag_use_validation")    
    info("  + Validation Samples:   $n_valid") 
    info("  + Doubled:              $which")  
    info("=====================================")

    # Scale the learning rate by the batch size
    lr=lr/batch_size

    # Random initialization of the persistent chains
    rbmaux.persistent_chain_vis,_ = random_columns(X,batch_size)
    rbmaux.persistent_chain_hid   = ProbHidCondOnVis(rbmaux, rbmaux.persistent_chain_vis)

    use_persistent = false
    for itr=1:n_iter
        # Check to see if we can use persistence at this epoch
        use_persistent = itr>=persistent_start ? persistent : false

        tic()

        # Mini-batch fitting loop. 
        @showprogress 1 "Fitting Batches..." for i=1:n_batches
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            batch = full(batch)
          
            fit_batch!(rbmaux, batch; persistent=use_persistent, 
                                   NormalizationApproxIter=NormalizationApproxIter,
                                   weight_decay=weight_decay,
                                   decay_magnitude=decay_magnitude,
                                   lr=lr, approx=approx)
            # tying up the weights
            if which=="input"
                rbmaux.W[:,n_features+1:end]=rbmaux.W[:,1:n_features]
                rbmaux.W2[:,n_features+1:end]=rbmaux.W2[:,1:n_features]
                rbmaux.W3[:,n_features+1:end]=rbmaux.W3[:,1:n_features]
                rbmaux.vbias[n_features+1:end]=rbmaux.vbias[1:n_features]
            elseif which=="output"
                rbmaux.W[n_hidden+1:end,:]=rbmaux.W[1:n_hidden,:]
                rbmaux.W2[n_hidden+1:end,:]=rbmaux.W2[1:n_hidden,:]
                rbmaux.W3[n_hidden+1:end,:]=rbmaux.W3[1:n_hidden,:]
                rbmaux.hbias[n_hidden+1:end]=rbmaux.hbias[1:n_hidden]
            end
        end
        
        # Get the average wall-time in µs
        walltime_µs=(toq()/n_batches/N)*1e6

        UpdateMonitor!(rbmaux,ProgressMonitor,X,itr;bt=walltime_µs,validation=validation)
        ShowMonitor(rbmaux,ProgressMonitor,X,itr)
    end

    # Passing result of training to original rbm
    if which=="input"
        rbm.W=rbmaux.W[:,1:n_features]
        rbm.W2=rbmaux.W2[:,1:n_features]
        rbm.W3=rbmaux.W3[:,1:n_features]
        rbm.vbias=rbmaux.vbias[1:n_features]
        rbm.hbias=rbmaux.hbias
        rbm.persistent_chain_vis=rbmaux.persistent_chain_vis[1:n_features,:]
        rbm.persistent_chain_hid=rbmaux.persistent_chain_hid
    elseif which=="output"
        rbm.W=rbmaux.W[1:n_hidden,:]
        rbm.W2=rbmaux.W2[1:n_hidden,:]
        rbm.W3=rbmaux.W3[1:n_hidden,:]
        rbm.vbias=rbmaux.vbias
        rbm.hbias=rbmaux.hbias[1:n_hidden]
        rbm.persistent_chain_vis=rbmaux.persistent_chain_vis
        rbm.persistent_chain_hid=rbmaux.persistent_chain_hid[1:n_hidden,:]
    end
    return rbm, ProgressMonitor
end

function pre_fit(dbm::DBM, X::Mat{Float64};
             persistent=true, lr=0.1, n_iter=10, batch_size=100, NormalizationApproxIter=1,
             weight_decay="none",decay_magnitude=0.01,validation=[],
             monitor_every=5,monitor_vis=false, approx="CD",
             persistent_start=1)

    depth = length(dbm)
    # if using the training method introduced by Salakhutdinov&Hinton mixing CD and MF, pretraining should be done using CD.
    approx = approx == "mixed" ? "CD" : approx

    rbm, _ = fit_doubled(dbm[1], X, "input";
             persistent=persistent, lr=lr, n_iter=n_iter, batch_size=batch_size, NormalizationApproxIter=NormalizationApproxIter,
             weight_decay=weight_decay,decay_magnitude=decay_magnitude,validation=validation,
             monitor_every=monitor_every,monitor_vis=monitor_vis, approx=approx, persistent_start=persistent_start)
    layers =  [("vishid1", rbm)]

    vis = X
    hid, _ = sample_hiddens(dbm[1], 2.*vis)  # double input to activate the hiddens
    for l=2:(depth-1)
        vis = hid
        println(dbm[l])
        rbm, _ = fit(dbm[l],vis;persistent=persistent, lr=lr, n_iter=n_iter, batch_size=batch_size, NormalizationApproxIter=NormalizationApproxIter,
             weight_decay=weight_decay,decay_magnitude=decay_magnitude,validation=validation,
             monitor_every=monitor_every,monitor_vis=monitor_vis, approx=approx, persistent_start=persistent_start)
        hid, _ = sample_hiddens(rbm, vis)
        rbm.W=rbm.W./2
        rbm.W2=rbm.W.*rbm.W
        rbm.W3=rbm.W2.*rbm.W
        rbm.vbias=layers[l-1][2].hbias
        rbm.hbias=rbm.hbias./2
        push!(layers,("hid$(l-1)hid$l",rbm))
    end

    vis = hid
    rbm, _ = fit_doubled(dbm[depth], vis,"output";persistent=persistent, lr=lr, n_iter=n_iter, batch_size=batch_size, NormalizationApproxIter=NormalizationApproxIter,
             weight_decay=weight_decay,decay_magnitude=decay_magnitude,validation=validation,
             monitor_every=monitor_every,monitor_vis=monitor_vis, approx=approx, persistent_start=persistent_start)
    push!(layers,("hid$(depth-1)hid$depth",rbm))
   
    dbm = DBM(layers)

    return dbm
end
