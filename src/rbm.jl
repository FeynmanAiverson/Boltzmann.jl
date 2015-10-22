
using Distributions
using ProgressMeter
using Base.LinAlg.BLAS
using Compat
using Devectorize
using HDF5
using PyCall
@pyimport matplotlib.pyplot as plt
@pyimport numpy as np


import Base.getindex
import StatsBase.fit

typealias Mat{T} AbstractArray{T, 2}
typealias Vec{T} AbstractArray{T, 1}

typealias Gaussian Normal

abstract AbstractRBM
abstract AbstractMonitor

@runonce type RBM{V,H} <: AbstractRBM
    W::Matrix{Float64}
    W2::Matrix{Float64}
    W3::Matrix{Float64}
    vbias::Vector{Float64}
    hbias::Vector{Float64}
    dW::Matrix{Float64}
    dW_prev::Matrix{Float64}
    persistent_chain_vis::Matrix{Float64}
    persistent_chain_hid::Matrix{Float64}
    momentum::Float64
    VisShape::Tuple{Int,Int}
end

@runonce type Monitor <: AbstractMonitor
    LastIndex::Int
    UseValidation::Bool
    MonitorEvery::Int
    MonitorVisual::Bool
    MonitorText::Bool
    Epochs::Vector{Float64}
    LearnRate::Vector{Float64}
    Momentum::Vector{Float64}
    PseudoLikelihood::Vector{Float64}
    TAPLikelihood::Vector{Float64}
    ValidationPseudoLikelihood::Vector{Float64}
    ValidationTAPLikelihood::Vector{Float64}
    ReconError::Vector{Float64}
    ValidationReconError::Vector{Float64}
    BatchTime_µs::Vector{Float64}
    FigureHandle
end

function Monitor(n_iter,monitor_every;monitor_vis=false,monitor_txt=true,validation=false)
    len = convert(Int,floor(n_iter/monitor_every))
    blank_vector1 = vec(fill!(Array(Float64,len,1),convert(Float64,NaN)))
    blank_vector2 = copy(blank_vector1)
    blank_vector3 = copy(blank_vector1)
    blank_vector4 = copy(blank_vector1)
    blank_vector5 = copy(blank_vector1)
    blank_vector6 = copy(blank_vector1)
    blank_vector7 = copy(blank_vector1)
    blank_vector8 = copy(blank_vector1)
    blank_vector9 = copy(blank_vector1)
    blank_vector10 = copy(blank_vector1)

    if monitor_vis
        fh = plt.figure(1;figsize=(11,15))
    else
        fh = NaN
    end

    Monitor(0,                   # Last Index
            validation,          # Flag for validation set
            monitor_every,       # When to display
            monitor_vis,         # Monitor visal display flag
            monitor_txt,         # Monitor text display flag
            blank_vector1,       # Epochs (for x-axes)
            blank_vector2,       # Learn Rate
            blank_vector3,       # Momentum
            blank_vector4,       # Pseudo-Likelihood
            blank_vector5,       # Tap-Likelihood
            blank_vector6,       # Validation Pseudo-Likelihood
            blank_vector7,       # Validation  Tap-Likelihood
            blank_vector8,       # ReconError
            blank_vector9,       # ValidationReconError
            blank_vector10,      # BatchTime_µs
            fh)                  # Monitor Figure Handle
end

function UpdateMonitor!(rbm::RBM,mon::Monitor,dataset::Mat{Float64},itr::Int;validation=[],bt=NaN,lr=NaN,mo=NaN)
    nh = size(rbm.W,1)
    nv = size(rbm.W,2)
    N = nh + nv
    nsamps = min(size(dataset,2),5000)      # Set maximum number of samples to test as 5000


    if itr%mon.MonitorEvery==0
        if mon.UseValidation 
            vpl = mean(score_samples(rbm, validation))/N
            vtl = mean(score_samples_TAP(rbm, validation))/N
            vre = recon_error(rbm,validation)/N
        else
            vpl = NaN
            vtl = NaN
            vre = NaN
        end
        pl = mean(score_samples(rbm, dataset[:,1:nsamps]))/N  
        tl = mean(score_samples_TAP(rbm, dataset[:,1:nsamps]))/N    
        re = recon_error(rbm,dataset[:,1:nsamps])/N

        mon.LastIndex+=1
        li = mon.LastIndex

        mon.PseudoLikelihood[li] = pl
        mon.TAPLikelihood[li] = tl
        mon.ReconError[li] = re
        mon.ValidationPseudoLikelihood[li] = vpl
        mon.ValidationTAPLikelihood[li] = vtl
        mon.ValidationReconError[li] = vre
        mon.Epochs[li] = itr
        mon.Momentum[li] = mo
        mon.LearnRate[li] = lr
        mon.BatchTime_µs[li] = bt
    end 
end


function SaveMonitorh5(mon::Monitor,filename::AbstractString)
    h5open(filename , "w") do file
        # write(file, "LastIndex", mon.LastIndex)
        # write(file, "UseValidation", mon.UseValidation)
        write(file, "MonitorEvery", mon.MonitorEvery)
        # write(file, "MonitorVisual", mon.MonitorVisual)
        # write(file, "MonitorText", mon.MonitorText)
        write(file, "Epochs", mon.Epochs)
        write(file, "LearnRate", mon.LearnRate)
        write(file, "Momentum", mon.Momentum)
        write(file, "PseudoLikelihood", mon.PseudoLikelihood)
        write(file, "TAPLikelihood", mon.TAPLikelihood)
        write(file, "ValidationPseudoLikelihood", mon.ValidationPseudoLikelihood)
        write(file, "ValidationTAPLikelihood", mon.ValidationTAPLikelihood)
        write(file, "ReconError", mon.ReconError)
        write(file, "ValidationReconError", mon.ValidationReconError)
        write(file, "BatchTime_µs", mon.BatchTime_µs)
        # write(file, "FigureHandle", mon.FigureHandle)
    end 
end  


function RBM(V::Type, H::Type,
             n_vis::Int, n_hid::Int,
             visshape::Tuple{Int,Int}; sigma=0.1, momentum=0.0, dataset=[])

    W = rand(Normal(0, sigma), (n_hid, n_vis))


    if isempty(dataset)
        RBM{V,H}(W,                                             # W
				 W.*W,							                # W2
				 W.*W.*W,							            # W3
                 zeros(n_vis),                                  # vbias
                 zeros(n_hid),                                  # hbias
                 zeros(n_hid, n_vis),                           # dW
                 zeros(n_hid, n_vis),                           # dW_prev
                 Array(Float64, 0, 0),                          # persistent_chain_vis
                 Array(Float64, 0, 0),                          # persistent_chain_hid
                 momentum,                                      # momentum
                 visshape)                                      # Shape of the visible units (for display)
    else
        ProbVis = mean(dataset,2)   # Mean across samples
        ProbVis = max(ProbVis,1e-8)
        ProbVis = min(ProbVis,1 - 1e-8)
        @devec InitVis = log(ProbVis ./ (1-ProbVis))

     	RBM{V,H}(W,   		                                    # W
				 W.*W,							                # W2
				 W.*W.*W,							            # W3
                 vec(InitVis),                                  # vbias
                 zeros(n_hid),                                  # hbias
                 zeros(n_hid, n_vis),                           # dW
                 zeros(n_hid, n_vis),                           # dW_prev
                 Array(Float64, 0, 0),                          # persistent_chain_vis
                 Array(Float64, 0, 0),                          # persistent_chain_hid
                 momentum,                                      # momentum
                 visshape)                                      # Shape of the visible units (for display)
    end
end


function Base.show{V,H}(io::IO, rbm::RBM{V,H})
    n_vis = size(rbm.vbias, 1)
    n_hid = size(rbm.hbias, 1)
    print(io, "RBM{$V,$H}($n_vis, $n_hid)")
end


typealias BernoulliRBM RBM{Bernoulli, Bernoulli}
BernoulliRBM(n_vis::Int, n_hid::Int, visshape::Tuple{Int,Int}; sigma=0.1, momentum=0.0, dataset=[]) =
    RBM(Bernoulli, Bernoulli, n_vis, n_hid, visshape; sigma=sigma, momentum=momentum, dataset=dataset)
typealias GRBM RBM{Gaussian, Bernoulli}
GRBM(n_vis::Int, n_hid::Int, visshape::Tuple{Int,Int}; sigma=0.1, momentum=0.0, dataset=[]) =
    RBM(Gaussian, Bernoulli, n_vis, n_hid, visshape; sigma=sigma, momentum=momentum, dataset=dataset)


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
    return sample(H, means), means
end

function sample_visibles{V,H}(rbm::RBM{V,H}, hid::Mat{Float64})
    means = vis_means(rbm, hid)
    return sample(V, means)
end


function gibbs(rbm::RBM, vis::Mat{Float64}; n_times=1)
    #print("gibbs")
    v_pos = vis
    h_samp, h_pos = sample_hiddens(rbm, v_pos)
    h_neg = Array(Float64,0,0)::Mat{Float64}
    v_neg = Array(Float64,0,0)::Mat{Float64}
    if n_times > 0
    # Save computation by setting `n_times=0` in the case
    # of persistent CD.
        v_neg = sample_visibles(rbm, h_samp)
        h_samp, h_neg = sample_hiddens(rbm, v_neg)
        for i=1:n_times-1
            v_neg = sample_visibles(rbm, h_samp)
            h_samp, h_neg = sample_hiddens(rbm, v_neg)
        end
    end
    return v_pos, h_pos, v_neg, h_neg
end



### Base MF definitions
#### Naive mean field
function mag_vis_naive(rbm::RBM, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    return logistic(buf)
end    
# Defining a method with same arguments as other mean field approxiamtions
mag_vis_naive(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})=mag_vis_naive(rbm, m_hid) 

function mag_hid_naive(rbm::RBM, m_vis::Mat{Float64}) 
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    return logistic(buf)
end    

mag_hid_naive(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})=mag_hid_naive(rbm, m_vis) 
  
#### Second order development
function mag_vis_tap2(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    second_order = gemm('T', 'N', rbm.W2, m_hid-abs2(m_hid)).*(0.5-m_vis)
    axpy!(1.0, second_order, buf)
    return logistic(buf)
end  

function mag_hid_tap2(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    second_order = gemm('N', 'N', rbm.W2, m_vis-abs2(m_vis)).*(0.5-m_hid)
    axpy!(1.0, second_order, buf)
    return logistic(buf)
end

#### Third order development

function mag_vis_tap3(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    second_order = gemm('T', 'N', rbm.W2, m_hid-abs2(m_hid)).*(0.5-m_vis)
    third_order = gemm('T', 'N', rbm.W3, abs2(m_hid).*(1.-m_hid)).*(1/3-2*(m_vis-abs2(m_vis)))
    axpy!(1.0, second_order, buf)
    axpy!(1.0, third_order, buf)
    return logistic(buf)
end  

function mag_hid_tap3(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    second_order = gemm('N', 'N', rbm.W2, m_vis-abs2(m_vis)).*(0.5-m_hid)
    third_order = gemm('N', 'N', rbm.W3, abs2(m_vis).*(1.-m_vis)).*(1/3-2*(m_hid-abs2(m_hid)))
    axpy!(1.0, second_order, buf)
    axpy!(1.0, third_order, buf)
    return logistic(buf)
end



function iter_mag(rbm::RBM, vis::Mat{Float64}; n_times=3, approx="tap2")
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

function iter_mag_persist!(rbm::RBM, vis::Mat{Float64}; n_times=3, approx="tap2")
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

    m_vis = rbm.persistent_chain_vis
    m_hid = rbm.persistent_chain_hid

    for i=1:n_times
       m_vis = 0.5 * mag_vis(rbm, m_vis, m_hid) + 0.5 * m_vis
       m_hid = 0.5 * mag_hid(rbm, m_vis, m_hid) + 0.5 * m_hid
    end

    rbm.persistent_chain_vis = m_vis
    rbm.persistent_chain_hid = m_hid

    return v_pos, h_pos, m_vis, m_hid
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

function recon_error(rbm::RBM, vis::Mat{Float64})
    # Fully forward MF operation to get back to visible samples
    vis_rec = vis_means(rbm,hid_means(rbm,vis))
    # Get the total error over the whole tested visible set,
    # here, as MSE
    dif = vis_rec - vis
    mse = mean(dif.*dif)
    return mse
end

function score_samples_TAP(rbm::RBM, vis::Mat{Float64}; n_iter=30)
    _, _, m_vis, m_hid = iter_mag(rbm, vis; n_times=n_iter, approx="tap2")
    eps=1e-6
    m_vis = max(m_vis, eps)
    m_vis = min(m_vis, 1.0-eps)
    m_hid = max(m_hid, eps)
    m_hid = min(m_hid, 1.0-eps)

    m_vis2 = abs2(m_vis)
    m_hid2 = abs2(m_hid)

    S = - sum(m_vis.*log(m_vis)+(1.0-m_vis).*log(1.0-m_vis),1) - sum(m_hid.*log(m_hid)+(1.0-m_hid).*log(1.0-m_hid),1)
    U_naive = - gemv('T',m_vis,rbm.vbias)' - gemv('T',m_hid,rbm.hbias)' - sum(gemm('N','N',rbm.W,m_vis).*m_hid,1)
    Onsager = - 0.5 * sum(gemm('N','N',rbm.W2,m_vis-m_vis2).*(m_hid-m_hid2),1)    
    fe_tap = U_naive + Onsager - S
    fe = free_energy(rbm, vis)
    return fe_tap - fe
end 


function update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr; approx="CD")
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
    # rbm.W +=  dW
    axpy!(1.0, rbm.dW, rbm.W)
    rbm.W2=rbm.W.*rbm.W
    
    if approx == "tap3"
        rbm.W3=rbm.W2.*rbm.W
    end
    # save current dW
    copy!(rbm.dW_prev, rbm.dW)
end

function update_weights_QuadraticPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, decay_mag; approx="CD")
    # dW = zeros(size(rbm.W))
    
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

    # rbm.W += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)

    # Apply Weight-Decay Penalty
    # rbm.W += -lr * L2-Penalty-Gradient
    axpy!(-lr*decay_mag,rbm.W,rbm.dW)

    # rbm.W +=  dW
    axpy!(1.0, rbm.dW, rbm.W)
   
    rbm.W2=rbm.W.*rbm.W
    
    if approx == "tap3"
        rbm.W3=rbm.W2.*rbm.W
    end
    # save current dW
    copy!(rbm.dW_prev, rbm.dW)
end

function update_weights_LinearPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, decay_mag ; approx="CD")
    # dW = zeros(size(rbm.W))
    # dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, rbm.dW)          # Not flushing rbm.dW since we multiply w/ 0.0
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0,rbm.dW)

    if contains(approx,"tap") 
        buf2 = gemm('N', 'T', h_neg-abs2(h_neg), v_neg-abs2(v_neg)) .* rbm.W  
        axpy!(-lr, buf2, rbm.dW)
    end

    if approx == "tap3"
        buf3 = gemm('N','T', (h_neg-abs2(h_neg)) .* (0.5-h_neg), (v_neg-abs2(v_neg)) .* (0.5-v_neg)) .* rbm.W2
        axpy!(-2.0*lr, buf3, rbm.dW)  
    end  

    # rbm.W += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)

    # Apply Weight-Decay Penalty
    # rbm.W += -lr * L1-Penalty-Gradient
    axpy!(-lr*decay_mag,sign(rbm.W),rbm.dW)

    # rbm.W += lr * dW
    axpy!(1.0, rbm.dW, rbm.W)
    
    rbm.W2=rbm.W.*rbm.W
    
    if approx == "tap3"
        rbm.W3=rbm.W2.*rbm.W
    end
    # save current dW
    copy!(rbm.dW_prev, rbm.dW)
end


function contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int; approx="CD")
    if approx == "CD"     
        v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis; n_times=n_gibbs)
    else
        v_pos, h_pos, v_neg, h_neg = iter_mag(rbm, vis; n_times=n_gibbs, approx=approx)
    end    
    return v_pos, h_pos, v_neg, h_neg
end


function persistent_contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int; approx="CD")
    if size(rbm.persistent_chain_vis) != size(vis)
        # persistent_chain not initialized or batch size changed, re-initialize
        rbm.persistent_chain_vis = vis
        rbm.persistent_chain_hid = hid_means(rbm, vis)
    end

    if approx == "CD"
        # take positive samples from real data
        v_pos, h_pos, _, _ = gibbs(rbm, vis; n_times=1)
        # take negative samples from "fantasy particles"
        _, _, v_neg, h_neg = gibbs(rbm, rbm.persistent_chain_vis; n_times=n_gibbs)
        copy!(rbm.persistent_chain_vis,v_neg)
    else
        v_pos, h_pos, v_neg, h_neg = iter_mag_persist!(rbm, vis; n_times=n_gibbs, approx=approx)
    end    

    return v_pos, h_pos, v_neg, h_neg
end


function fit_batch!(rbm::RBM, vis::Mat{Float64};
                    persistent=true, lr=0.1, n_gibbs=1,
                    weight_decay="none",decay_magnitude=0.01, approx="CD")
    
    sampler = persistent ? persistent_contdiv : contdiv
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, n_gibbs; approx=approx)

    # Gradient Update on Weights
    if weight_decay=="l2"
        update_weights_QuadraticPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, decay_magnitude, approx=approx)
    elseif weight_decay=="l1"
        update_weights_LinearPenalty!(rbm, h_pos, v_pos, h_neg, v_neg, lr, decay_magnitude, approx=approx)
    else
        update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr, approx=approx)
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
             persistent=true, lr=0.1, n_iter=10, batch_size=100, n_gibbs=1,
             weight_decay="none",decay_magnitude=0.01,validation=[],
             monitor_every=5,monitor_vis=false, approx="CD",
             persistent_start=1)
#=
The core RBM training function. Learns the weights and biasings using 
either standard Contrastive Divergence (CD) or Persistent CD, depending on
the user options. 

### Required Inputs
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
    rbm.persistent_chain_hid = hid_means(rbm, rbm.persistent_chain_vis)

    use_persistent = false
    for itr=1:n_iter
        # Check to see if we can use persistence at this epoch
        use_persistent = itr>=persistent_start ? persistent : false

        tic()
        @showprogress 1 "Fitting Batches..." for i=1:n_batches
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            batch = full(batch)
          
            fit_batch!(rbm, batch; persistent=use_persistent, 
                                   n_gibbs=n_gibbs,
                                   weight_decay=weight_decay,
                                   decay_magnitude=decay_magnitude,
                                   lr=lr, approx=approx)
            
        end
        walltime_µs=(toq()/n_batches/N)*1e6
        
        UpdateMonitor!(rbm,ProgressMonitor,X,itr;bt=walltime_µs,validation=validation)
        ShowMonitor(rbm,ProgressMonitor,itr)
    end

    return rbm, ProgressMonitor
end