using Distributions
using Base.LinAlg.BLAS
using Devectorize
import Base.getindex
using Compat
# using RBMBase

# typealias Gaussian Normal
# abstract AbstractRBM
abstract Net

function Base.show(io::IO, net::Net)
    nettype = string(typeof(net))
    layer_str = join(net.layernames, ",")
    print(io, "$nettype($layer_str)")
end

"""
    # Boltzmann.DBM (DBMBase.jl)
    ## Description
        A structure for containing all of the Deep Boltzmann Machine (DBM)
        model parameters, described as a stack of RBMs. 
        Note that the l-th layer of hidden units is, in term of RBM vocabulary, 
        the l-th RBM 'visible' layer as well as the (l-1)-th RBM 'hidden' layer. 
        Thus, it is always ensured that neighboring RBMs share the same vector of biases for their common layer.
    ## Structure
        - `layers`
        - `layernames
"""
type DBM <: Net
    layers::Vector{RBM}
    layernames::Vector{AbstractString}
end

DBM{T<:@compat(Tuple{AbstractString,RBM})}(namedlayers::Vector{T}) =
    DBM(map(p -> p[2], namedlayers), map(p -> p[1], namedlayers))


"""
    # Boltzmann.PassVisHid2ToHid1  (DBMBase.jl)
    ## Description 
        Function computing the linear combination of 
        both bottom-up and top-down inputs for intermediate layers.
"""
function PassVisHid2ToHid1(rbm1::RBM, vis::Mat{Float64}, rbm2::RBM, hid2::Mat{Float64})
    return rbm2.W' * hid2 .+ rbm1.W * vis .+ rbm1.hbias
end

### These functions need to be generalized to detect the Distribution on 
### the hidden and visible variables.
"""
    # Boltzmann.ProbHidCondOnNeighbors  (DBMBase.jl)
"""
function ProbHidCondOnNeighbors(rbm1::RBM, vis::Mat{Float64}, rbm2::RBM, hid2::Mat{Float64})
    return logsig(PassVisHid2ToHid1(rbm1,vis,rbm2,hid2))
end


"""
    # Boltzmann.ProbHidAtLayerCondOnVis (DBMBase.jl)
    ## Description 
        Function performing naive MF approximate bottom-up inference up to specified layer
        TODO : add DBM specific augmented input 
"""

function ProbHidAtLayerCondOnVis(net::Net, vis::Mat{Float64}, layer::Int)
    hiddens = Array(Array{Float64, 2}, layer)
    hiddens[1] = ProbHidCondOnVis(net[1], vis)
    for k=2:layer
        hiddens[k] = ProbHidCondOnVis(net[k], hiddens[k-1])
    end
    hiddens[end]
end

"""
    # Boltzmann.ProbHidInitCondOnVis (DBMBase.jl)
    ## Description 
        Function performing naive MF approximate bottom-up inference
        returning complete array of hidden units marginals
        TODO : add DBM specific augmented input 
"""

function ProbHidInitCondOnVis(net::Net, vis::Mat{Float64})
    depth = length(net)
    array_hid_init = Array(Array{Float64}, depth)
    array_hid_init[1] = ProbHidCondOnVis(net[1], vis)
    for k=2:depth
        array_hid_init[k] = ProbHidCondOnVis(net[k], array_hid_init[k-1])
    end
    array_hid_init
end