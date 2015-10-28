using Distributions
using Base.LinAlg.BLAS

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
    means = ProbHidCondOnVis(rbm, vis)
    return sample(H, means), means
end

function sample_visibles{V,H}(rbm::RBM{V,H}, hid::Mat{Float64})
    means = ProbVisCondOnHid(rbm, hid)
    return sample(V, means)
end


function gibbs(rbm::RBM, vis::Mat{Float64}; n_times=1)
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