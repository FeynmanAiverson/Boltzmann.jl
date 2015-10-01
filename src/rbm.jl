
using Distributions
using Base.LinAlg.BLAS
using Compat
using Devectorize

import Base.getindex
import StatsBase.fit

typealias Mat{T} AbstractArray{T, 2}
typealias Vec{T} AbstractArray{T, 1}

typealias Gaussian Normal

abstract AbstractRBM

@runonce type RBM{V,H} <: AbstractRBM
    W::Matrix{Float64}
    vbias::Vector{Float64}
    hbias::Vector{Float64}
    dW::Matrix{Float64}
    dW_prev::Matrix{Float64}
    persistent_chain::Matrix{Float64}
    momentum::Float64
end

function RBM(V::Type, H::Type,
             n_vis::Int, n_hid::Int; sigma=0.01, momentum=0.0, dataset=[])

    if isempty(dataset)
        RBM{V,H}(rand(Normal(0, sigma), (n_hid, n_vis)),        # W
                 zeros(n_vis),                                  # vbias
                 zeros(n_hid),                                  # hbias
                 zeros(n_hid, n_vis),                           # dW
                 zeros(n_hid, n_vis),                           # dW_prev
                 Array(Float64, 0, 0),                          # persistent_chain
                 momentum)                                      # momentum
    else
        ProbVis = mean(dataset,2)   # Mean across samples
        ProbVis = max(ProbVis,1e-8)
        ProbVis = min(ProbVis,1 - 1e-8)
        @devec InitVis = log(ProbVis ./ (1-ProbVis))

        RBM{V,H}(rand(Normal(0, sigma), (n_hid, n_vis)),        # W                                         
             vec(InitVis),                                      # vbias         
             zeros(n_hid),                                      # hbias         
             zeros(n_hid, n_vis),                               # dW                 
             zeros(n_hid, n_vis),                               # dW_prev                 
             Array(Float64, 0, 0),                              # persistent_chain                 
             momentum)                                          # momentum     
    end
end


function Base.show{V,H}(io::IO, rbm::RBM{V,H})
    n_vis = size(rbm.vbias, 1)
    n_hid = size(rbm.hbias, 1)
    print(io, "RBM{$V,$H}($n_vis, $n_hid)")
end


typealias BernoulliRBM RBM{Bernoulli, Bernoulli}
BernoulliRBM(n_vis::Int, n_hid::Int; sigma=0.01, momentum=0.0, dataset=[]) =
    RBM(Bernoulli, Bernoulli, n_vis, n_hid; sigma=sigma, momentum=momentum, dataset=dataset)
typealias GRBM RBM{Gaussian, Bernoulli}
GRBM(n_vis::Int, n_hid::Int; sigma=0.01, momentum=0.0, dataset=[]) =
    RBM(Gaussian, Bernoulli, n_vis, n_hid; sigma=sigma, momentum=momentum, dataset=dataset)


### Base Definitions
function logistic(x::Mat{Float64})
    ## Using Devectorize Macro
    @devec s = 1 ./ (1 + exp(-x))
    return s
end

function logistic(x::Vec{Float64})
    ## Using Devectorize Macro
    @devec s = 1 ./ (1 + exp(-x))
    return s
end

function hid_means(rbm::RBM, vis::Mat{Float64})
    p = rbm.W * vis .+ rbm.hbias
    return logistic(p)
end

function vis_means(rbm::RBM, hid::Mat{Float64})
    p = rbm.W' * hid .+ rbm.vbias
    return logistic(p)
end

function sample(::Type{Bernoulli}, means::Mat{Float64})
    s = zeros(means)
    r = rand(size(means))
    @simd for i=1:length(means)
        @inbounds s[i] = r[i] < means[i] ? 1.0 : 0.0
    end

    return s
end

function sample(::Type{Gaussian}, means::Mat{Float64})
    sigma2 = 1                   # using fixed standard diviation
    samples = zeros(size(means))
    for j=1:size(means, 2), i=1:size(means, 1)
        samples[i, j] = rand(Normal(means[i, j], sigma2))
    end
    return samples
end
    
function sample_hiddens{V,H}(rbm::RBM{V, H}, vis::Mat{Float64})
    means = hid_means(rbm, vis)
    return sample(H, means)
end

function sample_visibles{V,H}(rbm::RBM{V,H}, hid::Mat{Float64})
    means = vis_means(rbm, hid)
    return sample(V, means)
end

function gibbs(rbm::RBM, vis::Mat{Float64}; n_times=1)
    v_pos = vis
    h_pos = sample_hiddens(rbm, v_pos)
    h_neg = Array(Float64,0,0)::Mat{Float64}
    v_neg = Array(Float64,0,0)::Mat{Float64}
    if n_times > 0
    # Save computation by setting `n_times=0` in the case
    # of persistent CD.
        v_neg = sample_visibles(rbm, h_pos)
        h_neg = sample_hiddens(rbm, v_neg)
        for i=1:n_times-1
            v_neg = sample_visibles(rbm, h_neg)
            h_neg = sample_hiddens(rbm, v_neg)
        end
    end

    return v_pos, h_pos, v_neg, h_neg
end


function free_energy(rbm::RBM, vis::Mat{Float64})
    vb = sum(vis .* rbm.vbias, 1)
    Wx_b_log = sum(log(1 + exp(rbm.W * vis .+ rbm.hbias)), 1)
    return - vb - Wx_b_log
end


function score_samples(rbm::RBM, vis::Mat{Float64}; sample_size=10000)
    if issparse(vis)
        # sparse matrices may be infeasible for this operation
        # so using only little sample
        cols = sample(1:size(vis, 2), sample_size)
        vis = full(vis[:, cols])
    end
    n_feat, n_samples = size(vis)
    vis_corrupted = copy(vis)
    idxs = rand(1:n_feat, n_samples)
    for (i, j) in zip(idxs, 1:n_samples)
        vis_corrupted[i, j] = 1 - vis_corrupted[i, j]
    end
    fe = free_energy(rbm, vis)
    fe_corrupted = free_energy(rbm, vis_corrupted)
    return n_feat * log(logistic(fe_corrupted - fe))
end


function update_weights!(rbm::RBM, 
                         h_pos::Mat{Float64}, v_pos::Mat{Float64}, 
                         h_neg::Mat{Float64}, v_neg::Mat{Float64}, 
                         lr::Float64)
    # rbm.dW = lr*(h_pos * v_pos') - lr*(h_neg * v_neg')
    gemm!('N', 'T', lr, h_neg, v_neg,  0.0, rbm.dW)          # Not flushing rbm.dW since we multiply w/ 0.0
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, rbm.dW)
    # rbm.dW += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)
    # rbm.W += rbm.dW
    axpy!(1.0, rbm.dW, rbm.W)
    # save current dW
    copy!(rbm.dW_prev, rbm.dW)
end

function update_weights_QuadraticPenalty!(rbm::RBM, 
                                          h_pos::Mat{Float64}, v_pos::Mat{Float64},
                                          h_neg::Mat{Float64}, v_neg::Mat{Float64}, 
                                          lr::Float64, decay_mag::Float64)
    # dW = lr*(h_pos * v_pos') - lr*(h_neg * v_neg')
    gemm!('N', 'T', lr, h_neg, v_neg,  0.0, rbm.dW)          # Not flushing rbm.dW since we multiply w/ 0.0
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, rbm.dW)
    # rbm.dW += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)
    
    # Apply Weight-Decay Penalty
    # rbm.W += -lr * L2-Penalty-Gradient
    axpy!(-lr*decay_mag,rbm.W,rbm.dW)

    # rbm.W +=  dW
    axpy!(1.0, rbm.dW, rbm.W)
    
    # save current dW
    copy!(rbm.dW_prev, rbm.dW)
end

function update_weights_LinearPenalty!(rbm::RBM, 
                                       h_pos::Mat{Float64}, v_pos::Mat{Float64},
                                       h_neg::Mat{Float64}, v_neg::Mat{Float64}, 
                                       lr::Float64, decay_mag::Float64)
    # dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, rbm.dW)          # Not flushing rbm.dW since we multiply w/ 0.0
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0,rbm.dW)

    # rbm.W += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)

    # Apply Weight-Decay Penalty
    # rbm.W += -lr * L1-Penalty-Gradient
    axpy!(-lr*decay_mag,sign(rbm.W),rbm.dW)

    # rbm.W += lr * dW
    axpy!(1.0, rbm.dW, rbm.W)
    
    # save current dW
    copy!(rbm.dW_prev, rbm.dW)
end


function contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int)
    v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis; n_times=n_gibbs)
    return v_pos, h_pos, v_neg, h_neg
end


function persistent_contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int)
    if size(rbm.persistent_chain) != size(vis)
        # If the persistent chain has not been set, initialize it 
        # to the visible samples. This should only happen for the first
        # training batch.
        rbm.persistent_chain = Array(Float64, size(vis,1), size(vis,2));
        copy!(rbm.persistent_chain,vis)
    end
    
    # take positive samples from real data
    v_pos, h_pos, _, _ = gibbs(rbm, vis; n_times=0)
    # take negative samples from "fantasy particles"
    _, _, v_neg, h_neg = gibbs(rbm, rbm.persistent_chain; n_times=n_gibbs)
    
    copy!(rbm.persistent_chain,v_neg)
    
    return v_pos, h_pos, v_neg, h_neg
end


function fit_batch!(rbm::RBM, vis::Mat{Float64};
                    persistent=true, lr=0.1, n_gibbs=1,
                    weight_decay="none",decay_magnitude=0.01)
    sampler = persistent ? persistent_contdiv : contdiv
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, n_gibbs)

    # Gradient Update on Weights
    if weight_decay=="l2"
        update_weights_QuadraticPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, decay_magnitude)
    elseif weight_decay=="l1"
        update_weights_LinearPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, decay_magnitude)
    else
        update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr)
    end

    axpy!(lr,vec(sum(h_pos-h_neg,2)),rbm.hbias)
    axpy!(lr,vec(sum(v_pos-v_neg,2)),rbm.vbias)
end

function transform(rbm::RBM, X::Mat{Float64})
    return hid_means(rbm, X)
end


function generate(rbm::RBM, vis::Vec{Float64}; n_gibbs=1)
    return gibbs(rbm, reshape(vis, length(vis), 1); n_times=n_gibbs)[3]
end

function generate(rbm::RBM, X::Mat{Float64}; n_gibbs=1)
    return gibbs(rbm, X; n_times=n_gibbs)[3]
end


function components(rbm::RBM; transpose=true)
    return if transpose rbm.W' else rbm.W end
end
# synonym
features(rbm::RBM; transpose=true) = components(rbm, transpose)

function fit(rbm::RBM, X::Mat{Float64};
             persistent=true, lr=0.1, n_iter=10, batch_size=100, n_gibbs=1,
             weight_decay="none",decay_magnitude=0.01,validation=[])
#=
The core RBM training function. Learns the weights and biasings using 
either standard Contrastive Divergence (CD) or Persistent CD, depending on
the user options. 

### Required Inputs
- *rbm:* RBM object, initialized by `RBM()`/`GRBM()`
- *X:* Set of training vectors

### Optional Inputs
 - *persistent:* Whether or not to use persistent-CD [default=true]
 - *n_iter:* Number of training epochs [default=10]
 - *batch_size:* Minibatch size [default=100]
 - *n_gibbs:* Number of Gibbs sampling steps on the Markov Chain [default=1]
 - *weight_decay:* A string value representing the regularization to add to apply to the 
                   weight magnitude during training {"none","l1","l2"}/ [default="none"]
 - *decay_magnitude:* Relative importance assigned to the weight regularization. Smaller
                      values represent less regularization. Should be in range (0,1). 
                      [default=0.01]
 - *validation:* An array of validation samples, e.g. a held out set of training data.
                 If passed, `fit` will also track generalization progress during training.
                 [default=empty-set]
=#
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


    # Print info to user
    m_ = rbm.momentum
    info("=====================================")
    info("RBM Training")
    info("=====================================")
    info("  + Training Samples:   $n_samples")
    info("  + Features:           $n_features")
    info("  + Hidden Units:       $n_hidden")
    info("  + Epochs to run:      $n_iter")
    info("  + Persistent CD?:     $persistent")
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

    for itr=1:n_iter
        tic()
        for i=1:n_batches
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            batch = full(batch)
            fit_batch!(rbm, batch; persistent=persistent, 
                                   n_gibbs=n_gibbs,
                                   weight_decay=weight_decay,
                                   decay_magnitude=decay_magnitude,
                                   lr=lr)
        end
        walltime=toq()/n_batches/N
        pl = mean(score_samples(rbm, X))/N
        if flag_use_validation
            pl_valid = mean(score_samples(rbm, validation))/N
            @printf("[Epoch %04d] Train(pl : %0.3f), Valid(pl : %0.3f)  [%0.3f µsec/batch/unit]\n",itr,pl,pl_valid,walltime*1e6)
        else
            @printf("[Epoch %04d] Train(pl : %0.3f)  [%0.3f sec/batch]\n",itr,pl,walltime)
        end
    end
    return rbm
end