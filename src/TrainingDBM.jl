
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


function fit_batch!(dbm::DBM, vis::Mat{Float64};
                    persistent=true, lr=0.1, NormalizationApproxIter=1,
                    weight_decay="none",decay_magnitude=0.01, approx="CD")
    
    # Determine how to acquire the positive samples based upon the persistence mode.
    # v_pos = vis
    # h_samples, h_pos = sample_hiddens(rbm,v_pos)
    # # Set starting points in teh case of persistence
    # if persistent
    #     if approx=="naive" || contains(approx,"tap")
    #         v_init = copy(rbm.persistent_chain_vis)      
    #         h_init = copy(rbm.persistent_chain_hid)       
    #     end
    #     if approx=="CD" 
    #         v_init = vis               # A dummy setting
    #         h_init,_ = sample_hiddens(rbm,rbm.persistent_chain_vis)
    #     end
    # else
    #     if approx=="naive" || contains(approx,"tap")
    #         v_init = vis
    #         h_init = h_pos
    #     end
    #     if approx=="CD"
    #         v_init = vis               # A dummy setting
    #         h_init = h_samples
    #     end
    # end        

    # # Calculate the negative samples according to the desired approximation mode
    # v_neg, h_neg = get_negative_samples(rbm,v_init,h_init,approx,NormalizationApproxIter)

    # # If we are in persistent mode, update the chain accordingly
    # if persistent
    #     copy!(rbm.persistent_chain_vis,v_neg)
    #     copy!(rbm.persistent_chain_hid,h_neg)
    # end

    # # Update on weights
    # calculate_weight_gradient!(rbm,h_pos,v_pos,h_neg,v_neg,lr,approx=approx)
    # if weight_decay == "l2"
    #     regularize_weight_gradient!(rbm,lr;L2Penalty=decay_magnitude)
    # end
    # if weight_decay == "l1"
    #     regularize_weight_gradient!(rbm,lr;L1Penalty=decay_magnitude)
    # end
    # update_weights!(rbm,approx)

    # # Gradient update on biases
    # rbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    # rbm.vbias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))

    return dbm
end


"""
    # Boltzmann.fit (training.jl)
    ## Function Call
        `fit(rbm::RBM, X::Mat{Float64}[, persistent, lr, batch_size, NormalizationApproxIter, weight_decay, 
                                         decay_magnitude, validation,monitor_ever, monitor_vis,
                                         approx, persistent_start])`
    ## Description
    The core RBM training function. Learns the weights and biasings using 
    either standard Contrastive Divergence (CD) or Persistent CD, depending on
    the user options. 
    
    - *rbm:* RBM object, initialized by `RBM()`/`GRBM()`
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
     - *::RBM* -- A trained RBM model.
     - *::Monitor* -- A Monitor structure containing information on the training progress over
                      epochs.
"""
function fit(rbm::RBM, X::Mat{Float64};
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

    # Check for the existence of a validation set
    flag_use_validation=false
    if length(validation)!=0
        flag_use_validation=true
        n_valid=size(validation,2)        
    end

    # Create the historical monitor
    ProgressMonitor = Monitor(n_iter,monitor_every;monitor_vis=monitor_vis,
                                                   validation=flag_use_validation)

    # Print info to user
    m_ = rbm.momentum
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
    info("=====================================")

    # Scale the learning rate by the batch size
    lr=lr/batch_size

    # Random initialization of the persistent chains
    rbm.persistent_chain_vis,_ = random_columns(X,batch_size)
    rbm.persistent_chain_hid = ProbHidCondOnVis(rbm, rbm.persistent_chain_vis)

    use_persistent = false
    for itr=1:n_iter
        # Check to see if we can use persistence at this epoch
        use_persistent = itr>=persistent_start ? persistent : false

        tic()

        # Mini-batch fitting loop. 
        @showprogress 1 "Fitting Batches..." for i=1:n_batches
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            batch = full(batch)
          
            fit_batch!(rbm, batch; persistent=use_persistent, 
                                   NormalizationApproxIter=NormalizationApproxIter,
                                   weight_decay=weight_decay,
                                   decay_magnitude=decay_magnitude,
                                   lr=lr, approx=approx)
            
        end
        
        # Get the average wall-time in µs
        walltime_µs=(toq()/n_batches/N)*1e6
        
        UpdateMonitor!(rbm,ProgressMonitor,X,itr;bt=walltime_µs,validation=validation)
        ShowMonitor(rbm,ProgressMonitor,X,itr)
    end

    return rbm, ProgressMonitor
end
