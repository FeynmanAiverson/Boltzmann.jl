
using Distributions
using Base.LinAlg.BLAS
using Compat
using AppleAccelerate
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
    dW_prev::Matrix{Float64}
    persistent_chain::Matrix{Float64}
    momentum::Float64
end

function RBM(V::Type, H::Type,
             n_vis::Int, n_hid::Int; sigma=0.01, momentum=0.5, dataset=[])

    if isempty(dataset)
        RBM{V,H}(rand(Normal(0, sigma), (n_hid, n_vis)),
                 zeros(n_vis), 
                 zeros(n_hid),
                 zeros(n_hid, n_vis),
                 Array(Float64, 0, 0),
                 momentum)
    else
        ProbVis = mean(dataset,2)   # Mean across samples
        ProbVis = max(ProbVis,1e-20)
        ProbVis = min(ProbVis,1 - 1e-20)
        @devec InitVis = log(ProbVis ./ (1-ProbVis))

        RBM{V,H}(rand(Normal(0, sigma), (n_hid, n_vis)),
             vec(InitVis), 
             zeros(n_hid),
             zeros(n_hid, n_vis),
             Array(Float64, 0, 0),
             momentum)
    end
end


function Base.show{V,H}(io::IO, rbm::RBM{V,H})
    n_vis = size(rbm.vbias, 1)
    n_hid = size(rbm.hbias, 1)
    print(io, "RBM{$V,$H}($n_vis, $n_hid)")
end


typealias BernoulliRBM RBM{Bernoulli, Bernoulli}
BernoulliRBM(n_vis::Int, n_hid::Int; sigma=0.001, momentum=0.9, dataset=[]) =
    RBM(Bernoulli, Bernoulli, n_vis, n_hid; sigma=sigma, momentum=momentum, dataset=dataset)
typealias GRBM RBM{Gaussian, Bernoulli}
GRBM(n_vis::Int, n_hid::Int; sigma=0.001, momentum=0.9, dataset=[]) =
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


### Apple Accelerate Definitions
function logisticAccel(x::Mat{Float64})
    s = AppleAccelerate.rec(1+AppleAccelerate.exp(-x))
    return s
end

function logisticAccel(x::Vec{Float64})
    s = AppleAccelerate.rec(1+AppleAccelerate.exp(-x))
    return s
end

function hid_meansAccel(rbm::RBM, vis::Mat{Float64})
    p = rbm.W * vis .+ rbm.hbias
    return logisticAccel(p)
end

function vis_meansAccel(rbm::RBM, hid::Mat{Float64})
    p = rbm.W' * hid .+ rbm.vbias
    return logisticAccel(p)
end

function sampleAccel(::Type{Bernoulli}, means::Mat{Float64})
    s = zeros(means)
    r = rand(size(means))
    @simd for i=1:length(means)
        @inbounds s[i] = r[i] < means[i] ? 1.0 : 0.0
    end    
    return s
end

function sampleAccel(::Type{Gaussian}, means::Mat{Float64})
    sigma2 = 1                   # using fixed standard diviation
    samples = zeros(size(means))
    for j=1:size(means, 2), i=1:size(means, 1)
        samples[i, j] = rand(Normal(means[i, j], sigma2))
    end
    return samples
end
    
function sample_hiddensAccel{V,H}(rbm::RBM{V, H}, vis::Mat{Float64})
    means = hid_meansAccel(rbm, vis)
    return sampleAccel(H, means)
end

function sample_visiblesAccel{V,H}(rbm::RBM{V,H}, hid::Mat{Float64})
    means = vis_meansAccel(rbm, hid)
    return sampleAccel(V, means)
end


function gibbs(rbm::RBM, vis::Mat{Float64}; n_times=1, accelerate=false, dorate=0.0)
    # To implement dropout, for this sampling stage, we need to
    # choose a set of hidden variables to suppress.
    #
    # According to §8.2 of (Srivastava 2014), the dropout needs to
    # happen on a per-sample basis, with a different dropout pattern
    # used for every sample in the mini-batch. Hence, we apply the
    # dropout within the Gibbs sampling stage.
    #
    # We need to remember that the visible units are being passed to
    # here as a matrix, so we need to generate a matrix of dropout patterns.
    suppressedUnits = rand(size(rbm.hbias,1),size(vis,2)) .< dorate   

    v_pos = vis
    if accelerate
        # If the user has specified the use of the AppleAccelerate framework,
        # call the optimized Accelerate versions of the sampler
        h_pos = sample_hiddensAccel(rbm, v_pos)
        h_pos[suppressedUnits] = 0.0                        # Apply Dropout
        v_neg = sample_visiblesAccel(rbm, h_pos)
        h_neg = sample_hiddensAccel(rbm, v_neg)
        h_neg[suppressedUnits] = 0.0                        # Apply Dropout
        for i=1:n_times-1
            v_neg = sample_visiblesAccel(rbm, h_neg)
            h_neg = sample_hiddensAccel(rbm, v_neg)
            h_neg[suppressedUnits] = 0.0                    # Apply Dropout
        end
    else        
        h_pos = sample_hiddens(rbm, v_pos)
        h_pos[suppressedUnits] = 0.0                        # Apply Dropout
        v_neg = sample_visibles(rbm, h_pos)
        h_neg = sample_hiddens(rbm, v_neg)
        h_neg[suppressedUnits] = 0.0                        # Apply Dropout
        for i=1:n_times-1
            v_neg = sample_visibles(rbm, h_neg)
            h_neg = sample_hiddens(rbm, v_neg)
            h_neg[suppressedUnits] = 0.0                    # Apply Dropout
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


function update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf)
    dW = buf
    # dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, dW)
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, dW)
    # rbm.dW += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, dW)
    # rbm.W += lr * dW
    axpy!(1.0, dW, rbm.W)
    # save current dW
    copy!(rbm.dW_prev, dW)
end

function update_weights_QuadraticPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf, decay_mag)
    dW = buf
    # dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, dW)
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, dW)

    # rbm.W += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, dW)

    # Apply Weight-Decay Penalty
    # rbm.W += -lr * L2-Penalty-Gradient
    axpy!(lr*decay_mag,-rbm.W,dW)

    # rbm.W += lr * dW
    axpy!(1.0, dW, rbm.W)
    
    # save current dW
    copy!(rbm.dW_prev, dW)
end

function update_weights_LinearPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf, decay_mag)
    dW = buf
    # dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, dW)
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, dW)

    # rbm.W += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, dW)

    # Apply Weight-Decay Penalty
    # rbm.W += -lr * L1-Penalty-Gradient
    axpy!(lr*decay_mag,-sign(rbm.W),dW)

    # rbm.W += lr * dW
    axpy!(1.0, dW, rbm.W)
    
    # save current dW
    copy!(rbm.dW_prev, dW)
end


function contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int; accelerate=false, dorate=0.0)
    v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis; n_times=n_gibbs, accelerate=accelerate, dorate=dorate)
    return v_pos, h_pos, v_neg, h_neg
end


function persistent_contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int; accelerate=false,dorate=0.0)
    if size(rbm.persistent_chain) != size(vis)
        # persistent_chain not initialized or batch size changed, re-initialize
        rbm.persistent_chain = vis
    end
    # take positive samples from real data
    v_pos, h_pos, _, _ = gibbs(rbm, vis;accelerate=accelerate,dorate=dorate)
    # take negative samples from "fantasy particles"
    rbm.persistent_chain, _, v_neg, h_neg = gibbs(rbm, vis; n_times=n_gibbs,accelerate=accelerate,dorate=dorate)
    return v_pos, h_pos, v_neg, h_neg
end


function fit_batch!(rbm::RBM, vis::Mat{Float64};
                    persistent=true, buf=None, lr=0.1, n_gibbs=1,accelerate=false,
                    weight_decay="none",decay_magnitude=0.01,dorate=0.0)
    
    buf = buf == None ? zeros(size(rbm.W)) : buf

    sampler = persistent ? persistent_contdiv : contdiv
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, n_gibbs; accelerate=accelerate, dorate=dorate)

    lr=lr/size(v_pos,2)

    # Gradient Update on Weights
    if weight_decay=="l2"
        update_weights_QuadraticPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf, decay_magnitude)
    elseif weight_decay=="l1"
        update_weights_LinearPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf, decay_magnitude)
    else
        update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf)
    end

    rbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    rbm.vbias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))
    return rbm
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
             persistent=true, lr=0.1, n_iter=10, batch_size=100, n_gibbs=1,accelerate=false,
             weight_decay="none",decay_magnitude=0.01, dorate=0.0)
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
 - *accelerate:* Flag controlling whether or not to use Apple's Accelerate framework
                 to speed up some computations. Unused on non-OSX systems. [default=true]
 - *weight_decay:* A string value representing the regularization to add to apply to the 
                   weight magnitude during training {"none","l1","l2"}/ [default="none"]
 - *decay_magnitude:* Relative importance assigned to the weight regularization. Smaller
                      values represent less regularization. Should be in range (0,1). 
                      [default=0.01]
 - *dorate:* Dropout-rate, specifies the percentage of hidden units which are dropped during
            the training procedure [default=0.0]
=#
    @assert minimum(X) >= 0 && maximum(X) <= 1

    # Check OS and deny AppleAccelerate to non-OSX systems
    accelerate = @osx? accelerate : false

    n_samples = size(X, 2)
    n_features = size(X, 1)
    n_batches = @compat Int(ceil(n_samples / batch_size))
    w_buf = zeros(size(rbm.W))
    pseudo_likelihood = zeros(n_iter,1)
    
    # Print info to user
    info("=====================================")
    info("RBM Training")
    info("=====================================")
    info("  + Training Samples:   $n_samples")
    info("  + Features:           $n_features")
    info("  + Epochs to run:      $n_iter")
    info("  + Persistent CD?:     $persistent")
    info("  + Learning rate:      $lr")
    info("  + Drop-out Rate (p):  $dorate")
    info("  + Gibbs Steps:        $n_gibbs")    
    info("=====================================")

    # Training Loop
    for itr=1:n_iter
        # Loop over mini-batches
        for i=1:n_batches
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            batch = full(batch)
            fit_batch!(rbm, batch, persistent=persistent;
                       buf=w_buf, n_gibbs=n_gibbs,dorate=dorate,weight_decay=weight_decay,decay_magnitude=decay_magnitude)
        end

        # Scoring
        this_pl = mean(score_samples(rbm, X))
        pseudo_likelihood[itr] = this_pl
        info("Iteration #$itr, pseudo-likelihood = $this_pl")
    end
    return rbm, pseudo_likelihood
end