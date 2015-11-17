
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

function calculate_weight_gradient!(rbm::RBM, h_pos::Mat{Float64}, v_pos::Mat{Float64}, h_neg::Mat{Float64}, v_neg::Mat{Float64}, lr::Float64; approx="CD")
    # dW = zeros(size(rbm.W))
    # dW = lr * ( (h_pos * v_pos') - (h_neg * v_neg') )
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, rbm.dW)
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, rbm.dW)

    if contains(approx,"tap") 
        buf2 = gemm('N', 'T', h_neg-abs2(h_neg), v_neg-abs2(v_neg)) .* rbm.W  
        axpy!(-lr, buf2, rbm.dW)
    end

    if approx == "tap3"
        buf3 = gemm('N','T', (h_neg-abs2(h_neg)) .* (0.5-h_neg), (v_neg-abs2(v_neg)) .* (0.5-v_neg)) .* rbm.W2
        axpy!(-2.0*lr, buf3, rbm.dW)  
    end    

    # rbm.dW += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)

    # save current dW
    # copy!(rbm.dW_prev, rbm.dW)
    ##------------------------------------------##
    # This content should be moved to the `fit_batch` function.
    # axpy!(1.0, rbm.dW, rbm.W)           # rbm.W +=  dW
    # rbm.W2=rbm.W.*rbm.W    
    # if approx == "tap3"
    #     rbm.W3=rbm.W2.*rbm.W
    # end
    ##------------------------------------------##
end

function update_weights!(rbm::RBM,approx::AbstractString)
    axpy!(1.0,rbm.dW,rbm.W)             # Take step: W = W + dW
    copy!(rbm.dW_prev, rbm.dW)          # Save the current step for future use
    if contains(approx,"tap")
        rbm.W2 = rbm.W  .* rbm. W       # Update Square [for EMF-TAP2]
    end
    if approx == "tap3"
        rbm.W3 = rbm.W2 .* rbm.W        # Update Cube   [for EMF-TAP3]
    end
 end

function regularize_weight_gradient!(rbm::RBM,LearnRate::Float64;L2Penalty::Float64=NaN,L1Penalty::Float64=NaN,DropOutRate::Float64=NaN)
    if !isnan(L2Penalty)
        axpy!(-LearnRate*L2Penalty,rbm.W,rbm.dW)
    end

    if !isnan(L1Penalty)
        axpy!(-LearnRate*L1Penalty,sign(rbm.W),rbm.dW)
    end

    if !isnan(DropOutRate)
        # Not yet implemented, so we do nothing.
        # TODO: Implement Drop-out, here.
    end
end

##-----------------------------------------------------------------------------------------------------------##
# This content should be depricated by shifting the regularization into the `fit_batch` function
# function update_weights_QuadraticPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, decay_mag; approx="CD")
#     # dW = zeros(size(rbm.W))
    
#     gemm!('N', 'T', lr, h_neg, v_neg, 0.0, rbm.dW)
#     gemm!('N', 'T', lr, h_pos, v_pos, -1.0, rbm.dW)

#     if contains(approx,"tap") 
#         buf2 = gemm('N', 'T', h_neg-abs2(h_neg), v_neg-abs2(v_neg)) .* rbm.W  
#         axpy!(-lr, buf2, rbm.dW)
#     end

#     if approx == "tap3"
#         buf3 = gemm('N','T', (h_neg-abs2(h_neg)) .* (0.5-h_neg), (v_neg-abs2(v_neg)) .* (0.5-v_neg)) .* rbm.W2
#         axpy!(-2.0*lr, buf3, rbm.dW)  
#     end  

#     # rbm.W += rbm.momentum * rbm.dW_prev
#     axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)

#     # Apply Weight-Decay Penalty
#     # rbm.W += -lr * L2-Penalty-Gradient
#     axpy!(-lr*decay_mag,rbm.W,rbm.dW)

#     # rbm.W +=  dW
#     axpy!(1.0, rbm.dW, rbm.W)
   
#     rbm.W2=rbm.W.*rbm.W
    
#     if approx == "tap3"
#         rbm.W3=rbm.W2.*rbm.W
#     end
#     # save current dW
#     copy!(rbm.dW_prev, rbm.dW)
# end

# function update_weights_LinearPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, decay_mag ; approx="CD")
#     # dW = zeros(size(rbm.W))
#     # dW = (h_pos * v_pos') - (h_neg * v_neg')
#     gemm!('N', 'T', lr, h_neg, v_neg, 0.0, rbm.dW)          # Not flushing rbm.dW since we multiply w/ 0.0
#     gemm!('N', 'T', lr, h_pos, v_pos, -1.0,rbm.dW)

#     if contains(approx,"tap") 
#         buf2 = gemm('N', 'T', h_neg-abs2(h_neg), v_neg-abs2(v_neg)) .* rbm.W  
#         axpy!(-lr, buf2, rbm.dW)
#     end

#     if approx == "tap3"
#         buf3 = gemm('N','T', (h_neg-abs2(h_neg)) .* (0.5-h_neg), (v_neg-abs2(v_neg)) .* (0.5-v_neg)) .* rbm.W2
#         axpy!(-2.0*lr, buf3, rbm.dW)  
#     end  

#     # rbm.W += rbm.momentum * rbm.dW_prev
#     axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)

#     # Apply Weight-Decay Penalty
#     # rbm.W += -lr * L1-Penalty-Gradient
#     axpy!(-lr*decay_mag,sign(rbm.W),rbm.dW)

#     # rbm.W += lr * dW
#     axpy!(1.0, rbm.dW, rbm.W)
    
#     rbm.W2=rbm.W.*rbm.W
    
#     if approx == "tap3"
#         rbm.W3=rbm.W2.*rbm.W
#     end
#     # save current dW
#     copy!(rbm.dW_prev, rbm.dW)
# end
##-----------------------------------------------------------------------------------------------------------##

# function contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int; approx="CD")
#     if approx == "CD"     
#         v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis; n_times=n_gibbs)
#     else
#         v_pos, h_pos, v_neg, h_neg = iter_mag(rbm, vis; n_times=n_gibbs, approx=approx)
#     end    
#     return v_pos, h_pos, v_neg, h_neg
# end


# function persistent_contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int; approx="CD")
#     if approx == "CD"
#         # take positive samples from real data
#         v_pos, h_pos, _, _ = gibbs(rbm, vis; n_times=1)
#         # take negative samples from "fantasy particles"
#         _, _, v_neg, h_neg = gibbs(rbm, rbm.persistent_chain_vis; n_times=n_gibbs)
#         copy!(rbm.persistent_chain_vis,v_neg)
#         copy!(rbm.persistent_chain_hid,h_neg)
#     else
#         v_pos, h_pos, v_neg, h_neg = iter_mag_persist!(rbm, vis; n_times=n_gibbs, approx=approx)
#     end    

#     return v_pos, h_pos, v_neg, h_neg
# end

function get_negative_samples(rbm::RBM,vis_init::Mat{Float64},hid_init::Mat{Float64},approx::AbstractString, iterations::Int)
    if approx=="naive" || contains(approx,"tap")
        v_neg, h_neg = equilibrate(rbm,vis_init,hid_init; iterations=iterations, approx=approx)
    end

    if approx=="CD"
        # In the case of Gibbs/MCMC sampling, we will take the binary visible samples as the negative
        # visible samples, and the expectation (means) for the negative hidden samples.
        v_neg, _, _, h_neg = MCMC(rbm, hid_init; iterations=iterations, StartMode="hidden")
    end

    return v_neg, h_neg
end


function fit_batch!(rbm::RBM, vis::Mat{Float64};
                    persistent=true, lr=0.1, n_gibbs=1,
                    weight_decay="none",decay_magnitude=0.01, approx="CD")
    
    # Determine how to acquire the positive samples based upon the persistence mode.
    v_pos = vis
    h_samples, h_pos = sample_hiddens(rbm,v_pos)
    # Set starting points in teh case of persistence
    if persistent
        if approx=="naive" || contains(approx,"tap")
            v_init = copy(rbm.persistent_chain_vis)      
            h_init = copy(rbm.persistent_chain_hid)       
        end
        if approx=="CD" 
            v_init = vis               # A dummy setting
            h_init,_ = sample_hiddens(rbm,rbm.persistent_chain_vis)
        end
    else
        if approx=="naive" || contains(approx,"tap")
            v_init = vis
            h_init = h_pos
        end
        if approx=="CD"
            v_init = vis               # A dummy setting
            h_init = h_samples
        end
    end        

    # Calculate the negative samples according to the desired approximation mode
    v_neg, h_neg = get_negative_samples(rbm,v_init,h_init,approx,n_gibbs)

    # If we are in persistent mode, update the chain accordingly
    if persistent
        copy!(rbm.persistent_chain_vis,v_neg)
        copy!(rbm.persistent_chain_hid,h_neg)
    end

    # Update on weights
    calculate_weight_gradient!(rbm,h_pos,v_pos,h_neg,v_neg,lr,approx=approx)
    if weight_decay == "l2"
        regularize_weight_gradient!(rbm,lr;L2Penalty=decay_magnitude)
    end
    if weight_decay == "l1"
        regularize_weight_gradient!(rbm,lr;L1Penalty=decay_magnitude)
    end
    update_weights!(rbm,approx)

    # Gradient update on biases
    rbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    rbm.vbias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))

    return rbm
end


"""
    # Boltzmann.fit (training.jl)
    ## Function Call
        `fit(rbm::RBM, X::Mat{Float64}[, persistent, lr, batch_size, n_gibbs, weight_decay, 
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
     - *n_gibbs:* Number of Gibbs sampling steps on the Markov Chain [default=1]
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
"""
function fit(rbm::RBM, X::Mat{Float64};
             persistent=true, lr=0.1, n_iter=10, batch_size=100, n_gibbs=1,
             weight_decay="none",decay_magnitude=0.01,validation=[],
             monitor_every=5,monitor_vis=false, approx="CD",
             persistent_start=1)

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
    info("  + Training Samples:   $n_samples")
    info("  + Features:           $n_features")
    info("  + Hidden Units:       $n_hidden")
    info("  + Epochs to run:      $n_iter")
    info("  + Persistent ?:       $persistent")
    info("  + Training approx:    $approx")
    info("  + Momentum:           $m_")
    info("  + Learning rate:      $lr")
    info("  + Gibbs Steps:        $n_gibbs")   
    info("  + Weight Decay?:      $weight_decay") 
    info("  + Weight Decay Mag.:  $decay_magnitude")
    info("  + Validation Set?:    $flag_use_validation")    
    info("  + Validation Samples: $n_valid")   
    info("=====================================")

    # Scale the learning rate by the batch size
    lr=lr/batch_size

    # Random initialization of the persistent chain
    # It is okay if it isn't used in the actual training procedure.
    p = shuffle!(collect(1:n_samples))[1:batch_size]
    rbm.persistent_chain_vis = Array(Float64,n_features,batch_size)
    for i=1:batch_size
        rbm.persistent_chain_vis[:,i] = X[:,p[i]]
    end
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
                                   n_gibbs=n_gibbs,
                                   weight_decay=weight_decay,
                                   decay_magnitude=decay_magnitude,
                                   lr=lr, approx=approx)
            
        end
        
        # Get the average wall-time in µs
        walltime_µs=(toq()/n_batches/N)*1e6
        
        UpdateMonitor!(rbm,ProgressMonitor,X,itr;bt=walltime_µs,validation=validation)
        ShowMonitor(rbm,ProgressMonitor,itr)
    end

    return rbm, ProgressMonitor
end
