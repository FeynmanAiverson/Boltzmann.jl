
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

function get_negative_samples(dbm::DBM, vis_init::Mat{Float64}, array_hid_init::Array{Array{Float64},1},approx::AbstractString, iterations::Int)
    if approx=="naive" || contains(approx,"tap")
        v_neg, array_h_neg = equilibrate(dbm,vis_init,array_hid_init; iterations=iterations, approx=approx)
    end

    if approx=="CD" 
        # In the case of Gibbs/MCMC sampling, we will take the binary visible samples as the negative
        # visible samples, and the expectation (means) for the negative hidden samples.
        # v_neg, _, _, h_neg = MCMC(rbm, hid_init; iterations=iterations, StartMode="hidden") ## TODO : implement MCMC for DBM
    end

    return v_neg, array_h_neg
end

function get_positive_samples(dbm::DBM, vis::Mat{Float64}, array_hid_init::Array{Array{Float64},1},approx::AbstractString, iterations::Int)
    if approx=="naive" || contains(approx,"tap")
        v_pos, array_h_pos = clamped_equilibrate(dbm,vis,array_hid_init; iterations=iterations, approx=approx)
    end

    if approx=="CD" ##TODO
        # In the case of Gibbs/MCMC sampling, we will take the binary visible samples as the negative
        # visible samples, and the expectation (means) for the negative hidden samples.
        # v_neg, _, _, h_neg = MCMC(rbm, hid_init; iterations=iterations, StartMode="hidden") ## TODO : implement MCMC for DBM
    end

    return v_pos, array_h_pos
end

function fit_batch!(dbm::DBM, vis::Mat{Float64};
                    persistent=true, lr=0.1, NormalizationApproxIter=1,
                    weight_decay="none",decay_magnitude=0.01, approx="CD")
    depth = length(dbm)-1

    if approx=="naive" || contains(approx,"tap")
        array_h_pos_init = ProbHidInitCondOnVis(dbm, vis)
        v_pos, array_h_pos = get_positive_samples(dbm, vis, array_h_pos_init, approx,NormalizationApproxIter)
        
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
    ## TODO : fix the contrastive divergence procedure      
    # elseif approx=="CD" 
    #     if persistent # Set starting points in the case of persistence
    #         v_pos = vis
    #         h_samples, h_pos = sample_hiddens(rbm,v_pos)
    #         v_init = vis               # A dummy setting
    #         h_init,_ = sample_hiddens(rbm,rbm.persistent_chain_vis)
    #     else
    #         v_init = vis               # A dummy setting
    #         h_init = h_samples
    #     end
    end        

    # Calculate the negative samples according to the desired approximation mode
    v_neg, array_h_neg = get_negative_samples(dbm,v_init,array_hid_init,approx,NormalizationApproxIter)

    # If we are in persistent mode, update the chain accordingly
    if persistent
        copy!(dbm[1].persistent_chain_vis,v_neg)
        for l=1:depth  
            copy!(dbm[l].persistent_chain_hid,array_h_neg[l])
        end
    end

    # Update on weights abd biases
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
        rbm.vbias = copy(dbm[l].hbias) ## Constraining biases of visible units to be equal to biases of hidden units from shallower layer
    end
    return dbm
end


# """
#     # Boltzmann.fit (training.jl)
#     ## Function Call
#         `fit(rbm::RBM, X::Mat{Float64}[, persistent, lr, batch_size, NormalizationApproxIter, weight_decay, 
#                                          decay_magnitude, validation,monitor_ever, monitor_vis,
#                                          approx, persistent_start])`
#     ## Description
#     The core RBM training function. Learns the weights and biasings using 
#     either standard Contrastive Divergence (CD) or Persistent CD, depending on
#     the user options. 
    
#     - *rbm:* RBM object, initialized by `RBM()`/`GRBM()`
#     - *X:* Set of training vectors

#     ### Optional Inputs
#      - *persistent:* Whether or not to use persistent-CD [default=true]
#      - *persistent_start:* At which epoch to start using the persistent chains. Only
#                            applicable for the case that `persistent=true`.
#                            [default=1]
#      - *lr:* Learning rate [default=0.1]
#      - *n_iter:* Number of training epochs [default=10]
#      - *batch_size:* Minibatch size [default=100]
#      - *NormalizationApproxIter:* Number of Gibbs sampling steps on the Markov Chain [default=1]
#      - *weight_decay:* A string value representing the regularization to add to apply to the 
#                        weight magnitude during training {"none","l1","l2"}. [default="none"]
#      - *decay_magnitude:* Relative importance assigned to the weight regularization. Smaller
#                           values represent less regularization. Should be in range (0,1). 
#                           [default=0.01]
#      - *validation:* An array of validation samples, e.g. a held out set of training data.
#                      If passed, `fit` will also track generalization progress during training.
#                      [default=empty-set]
#      - *score_every:* Controls at which epoch the progress of the fit is monitored. Useful to 
#                       speed up the fit procedure if detailed progress monitoring is not required.
#                       [default=5]

#     ## Returns
#      - *::RBM* -- A trained RBM model.
#      - *::Monitor* -- A Monitor structure containing information on the training progress over
#                       epochs.
# """
# function fit(rbm::RBM, X::Mat{Float64};
#              persistent=true, lr=0.1, n_iter=10, batch_size=100, NormalizationApproxIter=1,
#              weight_decay="none",decay_magnitude=0.01,validation=[],
#              monitor_every=5,monitor_vis=false, approx="CD",
#              persistent_start=1)

#     # TODO: This line needs to be changed to accomodate real-valued units
#     @assert minimum(X) >= 0 && maximum(X) <= 1

#     n_valid=0
#     n_features = size(X, 1)
#     n_samples = size(X, 2)
#     n_hidden = size(rbm.W,1)
#     n_batches = @compat Int(ceil(n_samples / batch_size))
#     N = n_hidden+n_features

#     # Check for the existence of a validation set
#     flag_use_validation=false
#     if length(validation)!=0
#         flag_use_validation=true
#         n_valid=size(validation,2)        
#     end

#     # Create the historical monitor
#     ProgressMonitor = Monitor(n_iter,monitor_every;monitor_vis=monitor_vis,
#                                                    validation=flag_use_validation)

#     # Print info to user
#     m_ = rbm.momentum
#     info("=====================================")
#     info("RBM Training")
#     info("=====================================")
#     info("  + Training Samples:     $n_samples")
#     info("  + Features:             $n_features")
#     info("  + Hidden Units:         $n_hidden")
#     info("  + Epochs to run:        $n_iter")
#     info("  + Persistent ?:         $persistent")
#     info("  + Training approx:      $approx")
#     info("  + Momentum:             $m_")
#     info("  + Learning rate:        $lr")
#     info("  + Norm. Approx. Iters:  $NormalizationApproxIter")   
#     info("  + Weight Decay?:        $weight_decay") 
#     info("  + Weight Decay Mag.:    $decay_magnitude")
#     info("  + Validation Set?:      $flag_use_validation")    
#     info("  + Validation Samples:   $n_valid")   
#     info("=====================================")

#     # Scale the learning rate by the batch size
#     lr=lr/batch_size

#     # Random initialization of the persistent chains
#     rbm.persistent_chain_vis,_ = random_columns(X,batch_size)
#     rbm.persistent_chain_hid = ProbHidCondOnVis(rbm, rbm.persistent_chain_vis)

#     use_persistent = false
#     for itr=1:n_iter
#         # Check to see if we can use persistence at this epoch
#         use_persistent = itr>=persistent_start ? persistent : false

#         tic()

#         # Mini-batch fitting loop. 
#         @showprogress 1 "Fitting Batches..." for i=1:n_batches
#             batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
#             batch = full(batch)
          
#             fit_batch!(rbm, batch; persistent=use_persistent, 
#                                    NormalizationApproxIter=NormalizationApproxIter,
#                                    weight_decay=weight_decay,
#                                    decay_magnitude=decay_magnitude,
#                                    lr=lr, approx=approx)
            
#         end
        
#         # Get the average wall-time in µs
#         walltime_µs=(toq()/n_batches/N)*1e6
        
#         UpdateMonitor!(rbm,ProgressMonitor,X,itr;bt=walltime_µs,validation=validation)
#         ShowMonitor(rbm,ProgressMonitor,X,itr)
#     end

#     return rbm, ProgressMonitor
# end
