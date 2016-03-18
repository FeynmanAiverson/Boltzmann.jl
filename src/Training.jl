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

function calculate_weight_gradient!(rbm::RBM, h_pos::Mat{Float64}, 
                                    v_pos::Mat{Float64}, h_neg::Mat{Float64}, 
                                    v_neg::Mat{Float64}, lr::Float64; 
                                    approx=:sampling)
    ## Load step buffer with negative-phase    
    # dW <- LearRate*<h_neg,v_neg>
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, rbm.dW)          
    ## Stubtract step buffer from positive-phase to get gradient    
    # dW <- LearnRate*<h_pos,v_pos> - dW
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, rbm.dW)         
    ## Second-Order EMF Correction (for EMF-TAP2, EMF-TAP3)
    if approx==:emf2 || approx==:emf3 
        buf2 = gemm('N', 'T', h_neg-abs2(h_neg), v_neg-abs2(v_neg)) .* rbm.W  
        axpy!(-lr, buf2, rbm.dW)
    end
    ## Third-Order EMF Correction (for EMF-TAP3)
    if approx==:emf3
        termOne = (h_neg-abs2(h_neg)) .* (0.5-h_neg)
        termTwo = (v_neg-abs2(v_neg)) .* (0.5-v_neg)
        buf3 = gemm('N','T', termOne, termTwo) .* rbm.W2
        axpy!(-2.0*lr, buf3, rbm.dW)  
    end    
    ## Apply Momentum (adding last gradient to this one)    
    # rbm.dW += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)    
end

function update_weights!(rbm::RBM,approx)
    axpy!(1.0,rbm.dW,rbm.W)             # Take step: W = W + dW
    copy!(rbm.dW_prev, rbm.dW)          # Save the current step for future use
    if approx==:emf2 || approx==:emf3 
        rbm.W2 = rbm.W  .* rbm. W       # Update Square [for EMF-TAP2]
    end
    if approx==:emf3 
        rbm.W3 = rbm.W2 .* rbm.W        # Update Cube   [for EMF-TAP3]
    end
 end

function regularize_weight_gradient!(rbm::RBM,LearnRate::Float64;
                                     L2Penalty::Float64=NaN,
                                     L1Penalty::Float64=NaN,
                                     DropOutRate::Float64=NaN)
    ## Quadratic penalty on weights (Energy shrinkage)
    if !isnan(L2Penalty)
        axpy!(-LearnRate*L2Penalty,rbm.W,rbm.dW)
    end
    ## Linear penalty on weights (Sparsifying)
    if !isnan(L1Penalty)
        axpy!(-LearnRate*L1Penalty,sign(rbm.W),rbm.dW)
    end
    ## Dropout Regularization (restricted set of updates)
    if !isnan(DropOutRate)
        # Not yet implemented, so we do nothing.
        # TODO: Implement Drop-out, here.
    end
end

function get_negative_samples(rbm::RBM, vis_init::Mat{Float64},
                              hid_init::Mat{Float64}, approx, 
                              iterations::Int)
    if approx==:emf1 || approx==:emf2 || approx==:emf3
        v_neg, h_neg = equilibrate(rbm,vis_init,hid_init; 
                                   iterations=iterations, approx=approx)
    end

    if approx==:sampling
        # In the case of Gibbs/mcmc sampling, we will take the binary visible 
        # samples as the negative visible samples, and the expectation (means) 
        # for the negative hidden samples.
        v_neg, _, _, h_neg = mcmc(rbm, hid_init; 
                                  iterations=iterations, StartMode="hidden")
    end

    return v_neg, h_neg
end

function generate(rbm::RBM, vis_init::Mat{Float64},
                  approx, SamplingIterations::Int)
    Nsamples = size(vis_init,2)
    Nhid     = size(rbm.hbias,1)
    h_init  = zeros(Nsamples,Nhid)

    if approx==:emf1 || approx==:emf2 || approx==:emf3
        _, hid_mag = equilibrate(rbm,vis_init,hid_init; 
                                 iterations=SamplingIterations, approx=approx)
    end

    if approx==:sampling
        _, hid_mag, _, _ = mcmc(rbm, vis_init; 
                                iterations=SamplingIterations, 
                                StartMode="visible")
    end

    samples,_ = sample_visibles(rbm,hid_mag)

    return reshape(samples,rbm.VisShape)
end

function fit_batch!(rbm::RBM, vis::Mat{Float64}, opts::Dict;
                    persistent=nothing, lr=nothing)
    # If a different learning rate isn't specified, we use the one in the 
    # the options dictionary.
    if lr==nothing
      lr = opts[:learnRate]
    end
    if persistent==nothing
      persistent = opts[:persist]
    end


    # Determine how to acquire the positive samples based upon 
    # the persistence mode.
    v_pos = vis
    h_samples, h_pos = sample_hiddens(rbm,v_pos)
    # Set starting points in teh case of persistence
    if persistent
        if opts[:approxType]==:emf1 || 
           opts[:approxType]==:emf2 || opts[:approxType]==:emf3
            v_init = copy(rbm.persistent_chain_vis)      
            h_init = copy(rbm.persistent_chain_hid)       
        end
        if opts[:approxType]==:sampling
            v_init = vis               # A dummy setting
            h_init,_ = sample_hiddens(rbm,rbm.persistent_chain_vis)
        end
    else
        if opts[:approxType]==:emf1 || 
           opts[:approxType]==:emf2 || opts[:approxType]==:emf3
            v_init = vis
            h_init = h_pos
        end
        if opts[:approxType]==:sampling
            v_init = vis               # A dummy setting
            h_init = h_samples
        end
    end        

    # Calculate the negative samples according to the desired approximation mode
    v_neg, h_neg = get_negative_samples(rbm,v_init,h_init,
                                        opts[:approxType],
                                        opts[:approxIters])

    # If we are in persistent mode, update the chain accordingly
    if persistent
        copy!(rbm.persistent_chain_vis,v_neg)
        copy!(rbm.persistent_chain_hid,h_neg)
    end

    # Update on weights
    calculate_weight_gradient!(rbm,h_pos,v_pos,h_neg,v_neg,lr,
                               approx=opts[:approxType])
    if opts[:weightDecayType] == :l2
        regularize_weight_gradient!(rbm,lr;
                                    L2Penalty=opts[:weightDecayMagnitude])
    end
    if opts[:weightDecayType] == :l1
        regularize_weight_gradient!(rbm,lr;
                                    L1Penalty=opts[:weightDecayMagnitude])
    end
    update_weights!(rbm,opts[:approxType])

    # Gradient update on biases
    rbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    rbm.vbias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))

    return rbm
end


"""
    # Boltzmann.fit (training.jl)
    ## Function Call
        `fit(rbm::RBM, X::Mat{Float64}[, persistent, lr, opts[:batchSize], NormalizationApproxIter, weight_decay, 
                                         decay_magnitude, validation,monitor_ever, monitor_vis,
                                         approx, persistent_start, save_params])`
    ## Description
    The core RBM training function. Learns the weights and biasings using 
    either standard Contrastive Divergence (CD) or Persistent CD, depending on
    the user opts. 
    
    - *rbm:* RBM object, initialized by `RBM()`/`GRBM()`
    - *X:* Set of training vectors

    ### Optional Inputs
     - *persistent:* Whether or not to use persistent-CD [default=true]
     - *persistent_start:* At which epoch to start using the persistent chains. Only
                           applicable for the case that `persistent=true`.
                           [default=1]
     - *lr:* Learning rate [default=0.1]
     - *n_iter:* Number of training epochs [default=10]
     - *opts[:batchSize]:* Minibatch size [default=100]
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
     - *save_progress:* Controls the saving of RBM parameters throughout the course of the training.
                     Should be passed as a tuple in the following manner:
                        (::AbstractString,::Int)                      
                     where the first field is the filename for the HDF5 used to save results and
                     the second field specifies how often to write the parameters to disk. All
                     results will be stored in the specified HDF5 file under the root headings
                        `Epochxxxx___weight`
                        `Epochxxxx___vbias`
                        `Epochxxxx___bias`
                     where `xxxx` specifies the epoch number in the `%04d` format.   
                     [default=nothing]    

    ## Returns
     - *::RBM* -- A trained RBM model.
     - *::Monitor* -- A Monitor structure containing information on the training progress over
                      epochs.
"""
function fit(rbm::RBM, X::Mat{Float64}, opts::Dict)    
    # Copy user opts onto the default dictionary
    opts = dictionary_union(default_train_parameters(),opts)
    require_parameter(opts,:learnRate)
    require_parameter(opts,:batchSize) 
    # Force learning rate scaling, here
    scaledLearningRate  = opts[:learnRate] / opts[:batchSize]

    # TODO: This line needs to be changed to accomodate real-valued units
    @assert minimum(X) >= 0 && maximum(X) <= 1

    nValidation=0
    nFeatures = size(X, 1)
    nSamples = size(X, 2)
    nHidden = size(rbm.W,1)
    nBatches = @compat Int(ceil(nSamples / opts[:batchSize]))
    N = nHidden+nFeatures

    # Check for the existence of a validation set
    useValidation=false
    if !isempty(opts[:validationSet])
        useValidation=true
        nValidation=size(opts[:validationSet],2)        
    end

    # Create the historical monitor
    progressMonitor = Monitor(opts[:epochs],opts[:monitorEvery];
                              monitor_vis=opts[:monitorVis],
                              validation=useValidation)

    # Print info to user
    info("=====================================")
    info("RBM Training")
    info("=====================================")
    info("  + Training Samples:     $nSamples")
    info("  + Features:             $nFeatures")
    info("  + Hidden Units:         $nHidden")
    info("  + Epochs to run:        $(opts[:epochs])")
    info("  + Persistent ?:         $(opts[:persist])")
    info("  + Training approx:      $(opts[:approxType])")
    info("  + Momentum:             $(opts[:momentum])")
    info("  + Learning rate:        $(opts[:learnRate])")
    info("  + Norm. Approx. Iters:  $(opts[:approxIters])")   
    info("  + Weight Decay?:        $(opts[:weightDecayType])") 
    info("  + Weight Decay Mag.:    $(opts[:weightDecayMagnitude])")
    info("  + Validation Set?:      $useValidation")    
    info("  + Validation Samples:   $nValidation")   
    info("=====================================")

    # Random initialization of the persistent chains
    rbm.persistent_chain_vis,_ = random_columns(X,opts[:batchSize])
    rbm.persistent_chain_hid = condprob_hid(rbm, rbm.persistent_chain_vis)

    usePersistent = false
    for itr=1:opts[:epochs]
        # Check to see if we can use persistence at this epoch
        usePersistent = itr>=opts[:persistStart] ? opts[:persist] : false

        tic()

        # Mini-batch fitting loop. 
        @showprogress 1 "Fitting Batches..." for i=1:nBatches
            batch = X[:, ((i-1)*opts[:batchSize] + 1):min(i*opts[:batchSize], end)]
            batch = full(batch)
          
            fit_batch!(rbm, batch, opts; persistent=usePersistent, 
                                         lr=scaledLearningRate)
            
        end
        
        # Get the average wall-time in µs
        walltime_µs=(toq()/nBatches/N)*1e6
        
        update_monitor!(rbm,progressMonitor,X,itr;
                        bt=walltime_µs,
                        validation=opts[:validationSet],
                        lr=opts[:learnRate],
                        mo=opts[:momentum])
        show_monitor(rbm,progressMonitor,X,itr)

        # Attempt to save parameters if need be
        if itr%opts[:saveEvery]==0
            rootName = @sprintf("Epoch%04d",itr)
            if isfile(opts[:saveFile])
                info("Appending Params...")
                append_params(opts[:saveFile],rbm,rootName)
            else
                info("Creating file and saving params...")
                save_params(opts[:saveFile],rbm,rootName)
            end
        end
    end

    return rbm, progressMonitor
end


function default_train_parameters()
    D = Dict(:learnRate => nothing,
             :batchSize => nothing,
             :epochs => Integer(10),
             :approxType => :sample,
             :approxIters => Integer(1),
             :persist => false,
             :persistStart => Integer(1),
             :weightDecayType => :none,
             :weightDecayMagnitude => Float64(0.0),
             :validationSet => Array(Float64,0,0),
             :monitorEvery => Integer(1),
             :showMonitor => true,
             :saveEvery => Inf,
             :saveFile => AbstractString(""))
    return D    
end