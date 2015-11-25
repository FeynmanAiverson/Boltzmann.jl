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
        A structure for containing all of the restricted Boltzmann Machine (RBM)
        model parameters. Besides just the model parameters (couplings, biases),
        the structure also contains variables which are pertinent to the RBM training
        procedure.

    ## Structure
        - `layers`
        - `layernames
"""
type DBM <: Net
    layers::Vector{RBM}
    layernames::Vector{AbstractString}
end

DBM{T<:@compat(Tuple{AbstractString,RBM})}(namedlayers::Vector{T}) =
    DBN(map(p -> p[2], namedlayers), map(p -> p[1], namedlayers))

"""
    # Boltzmann.ProbHidLayerCondOnVis (DBMBase.jl)
"""

function ProbHidAtLayerCondOnVis(net::Net, vis::Mat{Float64}, layer::Int)
    hiddens = Array(Array{Float64, 2}, layer)
    hiddens[1] = ProbHidCondOnVis(net[1], vis)
    for k=2:layer
        hiddens[k] = ProbHidCondOnVis(net[k], hiddens[k-1])
    end
    hiddens[end]
end
