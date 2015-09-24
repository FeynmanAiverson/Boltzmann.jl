
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
    
function sample_hiddens{V,H}(rbm::RBM{V,H}, vis::Mat{Float64})
    means = hid_means(rbm, vis)
    return sample(H, means)
end

function sample_visibles{V,H}(rbm::RBM{V,H}, hid::Mat{Float64})
    means = vis_means(rbm, hid)
    return sample(V, means)
end

### Base MF definitions
#### Naive mean field
function mag_vis_naive(rbm::RBM, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    # buf = rbm.W'*m_hid.+rbm.vbias
    # buf = zeros(rbm.vbias)
    # copy!(buf,rbm.vbias)
    # gemm!('T', 'N', rbm.W, m_hid, 1.0, buf)
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    return logistic(buf)
end    

mag_vis_naive(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})=mag_vis_naive(rbm, m_hid) # Defining a method with same arguments as other mean field approxiamtions

function mag_hid_naive(rbm::RBM, m_vis::Mat{Float64}) 
    # buf = rbm.W*m_vis.+rbm.hbias
    # buf = zeros(rbm.hbias)
    # copy!(buf,rbm.hbias)  
    # gemm!('N', 'N', rbm.W, m_vis, 1.0, buf)
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    return logistic(buf)
end    

mag_hid_naive(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})=mag_hid_naive(rbm, m_vis) # Defining a method with same arguments as other mean field approxiamtio
  
#### Second order development

function mag_vis_tap2(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    # buf = rbm.W'*m_hid+rbm.vbias
    # buf = zeros(rbm.vbias)
    # copy!(buf,rbm.vbias)
    # gemm!('T', 'N', rbm.W, m_hid, 1.0, buf)
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    # \sum_j w_ij(m_j-m_jˆ2)(0.5-m_i)
    second_order = gemm('T', 'N', rbm.W.^2, m_hid-m_hid.^2).*(0.5-m_vis)
    axpy!(1.0, second_order, buf)
    return logistic(buf)
end  

function mag_hid_tap2(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    second_order = gemm('N', 'N', rbm.W.^2, m_vis-m_vis.^2).*(0.5-m_hid)
    axpy!(1.0, second_order, buf)
    return logistic(buf)
end

#### Third order development

function mag_vis_tap3(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    # buf = rbm.W'*m_hid+rbm.vbias
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    # \sum_j w_ijˆ2(m_j-m_jˆ2)(0.5-m_i)
    second_order = gemm('T', 'N', rbm.W.^2, m_hid-m_hid.^2).*(0.5-m_vis)
    # \sum_j w_ijˆ3(1/3-2(m_i-m-iˆ2))(m_jˆ2-m_jˆ3)
    third_order = gemm('T', 'N', rbm.W.^3, m_hid.^2-m_hid.^3).*(1/3-2*(m_vis-m_vis.^2))
    axpy!(1.0, second_order, buf)
    axpy!(1.0, third_order, buf)
    return logistic(buf)
end  

function mag_hid_tap3(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    second_order = gemm('N', 'N', rbm.W.^2, m_vis-m_vis.^2).*(0.5-m_hid)
    third_order = gemm('N', 'N', rbm.W.^3, m_vis.^2-m_vis.^3).*(1/3-2*(m_hid-m_hid.^2))
    axpy!(1.0, second_order, buf)
    axpy!(1.0, third_order, buf)
    return logistic(buf)
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


function gibbs(rbm::RBM, vis::Mat{Float64}; n_times=1, accelerate=false)
    v_pos = vis
    if accelerate
        # If the user has specified the use of the AppleAccelerate framework,
        # call the optimized Accelerate versions of the sampler
        h_pos = sample_hiddensAccel(rbm, v_pos)
        v_neg = sample_visiblesAccel(rbm, h_pos)
        h_neg = sample_hiddensAccel(rbm, v_neg)
        for i=1:n_times-1
            v_neg = sample_visiblesAccel(rbm, h_neg)
            h_neg = sample_hiddensAccel(rbm, v_neg)
        end
    else        
        h_pos = sample_hiddens(rbm, v_pos)
        v_neg = sample_visibles(rbm, h_pos)
        h_neg = sample_hiddens(rbm, v_neg)
        for i=1:n_times-1
            v_neg = sample_visibles(rbm, h_neg)
            h_neg = sample_hiddens(rbm, v_neg)
        end
    end
    return v_pos, h_pos, v_neg, h_neg
end

function iter_mag(rbm::RBM, vis::Mat{Float64}; n_times=1, approx="tap2")
    v_pos = vis
    h_pos = hid_means(rbm, v_pos)
    if approx == "naive"
        mag_vis = mag_vis_naive
        mag_hid = mag_hid_naive
    elseif approx == "tap3"
        mag_vis = mag_vis_tap3
        mag_hid = mag_hid_tap3
    else    
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    end    

    m_vis = 0.5 * mag_vis(rbm, vis, h_pos) + 0.5 * vis
    m_hid = 0.5 * mag_hid(rbm, m_vis, h_pos) + 0.5 * h_pos
    for i=1:n_times-1
       m_vis = 0.5 * mag_vis(rbm, m_vis, m_hid) + 0.5 * m_vis
       m_hid = 0.5 * mag_hid(rbm, m_vis, m_hid) + 0.5 * m_hid
    end
    return v_pos, h_pos, m_vis, m_hid
end    

 # m_vis = mag_vis_naive(rbm, h_pos)
 #    m_hid = mag_hid_naive(rbm, m_vis)
 #    for i=1:n_times-1
 #        m_vis = mag_vis_naive(rbm, m_hid)
 #        m_hid = mag_hid_naive(rbm, m_vis)


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

# for 
function update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf; approx="CD")
    dW = buf
    # dW = pos - neg
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, dW)
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, dW)

    #println("first order term   ",sum(dW)/(size(dW,1)*size(dW,2)))
    if contains(approx,"tap") 
        buf2 = gemm('N', 'T', h_neg-h_neg.^2, v_neg-v_neg.^2) .* rbm.W  
        axpy!(-lr, buf2, dW)
        #println("second order term  ",sum(buf2)/(size(dW,1)*size(dW,2)))
    end
    if approx == "tap3"
        buf3 = gemm('N','T', (h_neg-h_neg.^2) .* (0.5-h_neg), (v_neg-v_neg.^2) .* (0.5-v_neg)) .* rbm.W.^2
        axpy!(-2.0*lr, buf3, dW)  
        #println("third order term  ",sum(buf3)/(size(dW,1)*size(dW,2)))
    end    
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

    if contains(approx,"tap") 
        buf2 = gemm('N', 'T', h_neg-h_neg.^2, v_neg-v_neg.^2) .* rbm.W  
        axpy!(-lr, buf2, dW)
        #println("second order term  ",sum(buf2)/(size(dW,1)*size(dW,2)))
    end
    if approx == "tap3"
        buf3 = gemm('N','T', (h_neg-h_neg.^2) .* (0.5-h_neg), (v_neg-v_neg.^2) .* (0.5-v_neg)) .* rbm.W.^2
        axpy!(-2.0*lr, buf3, dW)  
        #println("third order term  ",sum(buf3)/(size(dW,1)*size(dW,2)))
    end 
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


function contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int; accelerate=false, approx="CD")
    if approx == "CD"     
        v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis; n_times=n_gibbs, accelerate=accelerate)
    else
        v_pos, h_pos, v_neg, h_neg = iter_mag(rbm, vis; n_times=n_gibbs, approx=approx)
    end    
    return v_pos, h_pos, v_neg, h_neg
end


function persistent_contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int; accelerate=false, approx="CD")
    if size(rbm.persistent_chain) != size(vis)
        # persistent_chain not initialized or batch size changed, re-initialize
        rbm.persistent_chain = vis
    end

    if approx == "CD"
        # take positive samples from real data
        v_pos, h_pos, _, _ = gibbs(rbm, vis; accelerate=accelerate)
        # take negative samples from "fantasy particles"
        _, _, v_neg, h_neg = gibbs(rbm, rbm.persistent_chain; n_times=n_gibbs,accelerate=accelerate)
        rbm.persistent_chain = v_neg
    else
        v_pos, h_pos, _, _ = iter_mag(rbm, vis; n_times=n_gibbs, approx=approx)
        _, _, v_neg, h_neg = iter_mag(rbm, rbm.persistent_chain; n_times=n_gibbs, approx=approx)
        rbm.persistent_chain = v_neg

    end    
    return v_pos, h_pos, v_neg, h_neg
end


function fit_batch!(rbm::RBM, vis::Mat{Float64};
                    persistent=true, buf=None, lr=0.1, n_gibbs=1,accelerate=false,
                    weight_decay="none",decay_magnitude=0.01, approx="CD")
    buf = buf == None ? zeros(size(rbm.W)) : buf

    sampler = persistent ? persistent_contdiv : contdiv
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, n_gibbs; accelerate=accelerate, approx=approx)

    lr=lr/size(v_pos,2)

    # Gradient Update on Weights
    if weight_decay=="l2"
        update_weights_QuadraticPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf, decay_magnitude)
    elseif weight_decay=="l1"
        update_weights_LinearPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf, decay_magnitude)
    else
        update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf, approx=approx)
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
             weight_decay="none",decay_magnitude=0.01, approx="CD")
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
=#
    @assert minimum(X) >= 0 && maximum(X) <= 1

    # Check OS and deny AppleAccelerate to non-OSX systems
    accelerate = @osx? accelerate : false

    n_samples = size(X, 2)
    n_batches = @compat Int(ceil(n_samples / batch_size))
    w_buf = zeros(size(rbm.W))
    for itr=1:n_iter
        # tic()
        for i=1:n_batches
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            batch = full(batch)
            fit_batch!(rbm, batch, persistent=persistent,
                       buf=w_buf, n_gibbs=n_gibbs, accelerate=accelerate;
                       weight_decay=weight_decay, decay_magnitude=decay_magnitude, approx=approx)
        end
        # toc()
        pseudo_likelihood = mean(score_samples(rbm, X))
        println("Iteration #$itr, pseudo-likelihood = $pseudo_likelihood")
    end
    return rbm
end