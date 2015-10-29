using Devectorize

if !isdefined(:__EXPRESSION_HASHES__)
    __EXPRESSION_HASHES__ = Set{UInt64}()
end

macro runonce(expr)
    h = hash(expr)
    return esc(quote
        if !in($h, __EXPRESSION_HASHES__)
            push!(__EXPRESSION_HASHES__, $h)
            $expr
        end
    end)
end

"""
  Defines the type `Mat{T}` as a convenience wrapper around the 
  `AbstractArray{T,2}` type. 
"""
typealias Mat{T} AbstractArray{T, 2}

"""
  Defines the type `Vec{T}` as a convenience wrapper around the 
  `AbstractArray{T,1}` type. 
"""
typealias Vec{T} AbstractArray{T, 1}

""" 
  # Boltzmann.normalize_samples (utils.jl)
  ## Function Calls
    `normalize_samples(X::Mat{Float64})`
  
  ## Description
    Given a matrix, `X`, assume that each column represents a different
    data sample and that each row represents a different feature. In this
    case, `normalize_samples` will normalize each individual sample to
    the range `[0,1]` according to the minimum and maximum features in the
    sample.

  ## Returns
    1. `::Mat{Float64}`, The normalized dataset.

  ### See also...
    `normalize_samples!`
"""
function normalize_samples(X::Mat{Float64})
    samples = size(X,2)

    for i=1:samples
      x = X[:,i]
      minx = minimum(x)
      maxx = maximum(x)
      ranx = maxx-minx
      X[:,i] = (x-minx)/ranx
    end

    return X
end

"""
  # Boltzmann.normalize_samples! (utils.jl)
  ## Function Calls
    `normalize_samples!(X::Mat{Float64})`
  
  ## Description
    Given a matrix, `X`, assume that each column represents a different
    data sample and that each row represents a different feature. In this
    case, `normalize_samples` will normalize each individual sample to
    the range `[0,1]` according to the minimum and maximum features in the
    sample. 

  ## Returns
    Nothing, modifies `X` in place.

  ### See also...
    `normalize_samples`
"""
function normalize_samples!(X::Mat{Float64})
    samples = size(X,2)

    for i=1:samples
      x = X[:,i]
      minx = minimum(x)
      maxx = maximum(x)
      ranx = maxx-minx
      X[:,i] = (x-minx)/ranx
    end
end

"""
  # Boltzmann.normalize (utils.jl)
  ## Function Calls
    `normalize(x::Mat{Float64})`
    `normalize(x::Vec{Float64})`
  
  ## Description
    Given an array, `x`, normalize the entire array to the range
    `[0,1]` according to the maximum and miniumum values of the array.

  ## Returns
    1. `::Mat{Float64}` *or* `::Vec{Float64}` depending on the input.

  ### See also...
    `normalize!`
"""
function normalize(x::Mat{Float64})
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds x[i] = (x[i]-minx) / ranx
    end

    return x
end
function normalize(x::Vec{Float64})
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds x[i] = (x[i]-minx) / ranx
    end

    return x
end

"""
  # Boltzmann.normalize! (utils.jl)
  ## Function Calls
    `normalize!(x::Mat{Float64})`
    `normalize!(x::Vec{Float64})`
  
  ## Description
    Given an array, `x`, normalize the entire array to the range
    `[0,1]` according to the maximum and miniumum values of the array.

  ## Returns
    Nothing. Modifies `x` in place.

  ### See also...
    `normalize`
"""
function normalize!(x::Mat{Float64})
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds x[i] = (x[i]-minx) / ranx
    end
end
function normalize!(x::Vec{Float64})
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds x[i] = (x[i]-minx) / ranx
    end
end


"""
  # Boltzmann.binarize (utils.jl)
  ## Function Calls
    `binarize(x::Mat{Float64}[,threshold=0.0])`
    `binarize(x::Vec{Float64}[,threshold=0.0])`
  
  ## Description
    Given an array, `x`, assign each element of the array to 
    either `0` or `1` depending on specified value of `threshold`.
    This is done according to the following rule...
    ```
        if element <= threshold: element = 0
        if element >  threshold: element = 1
    ```

  ## Returns
    1. `::Mat{Float64}` *or* `::Vec{Float64}`, depending on input.

  ### See also...
    `binarize!`
"""
function binarize(x::Mat{Float64};threshold=0.0)
  s = copy(x)
  @simd for i=1:length(x)
    @inbounds s[i] = x[i] > threshold ? 1.0 : 0.0
  end
  return s
end
function binarize(x::Vec{Float64};threshold=0.0)
  s = copy(x)
  @simd for i=1:length(x)
    @inbounds s[i] = x[i] > threshold ? 1.0 : 0.0
  end
  return s
end

"""
  # Boltzmann.binarize! (utils.jl)
  ## Function Calls
    `binarize!(x::Mat{Float64}[,threshold=0.0])`
    `binarize!(x::Vec{Float64}[,threshold=0.0])`
  
  ## Description
    Given an array, `x`, assign each element of the array to 
    either `0` or `1` depending on specified value of `threshold`.
    This is done according to the following rule...
    ```
        if element <= threshold: element = 0
        if element >  threshold: element = 1
    ```

  ## Returns
    Nothing. Modifies `x` in place.

  ### See also...
    `binarize`
"""
function binarize!(x::Mat{Float64};threshold=0.0)
  @simd for i=1:length(x)
    @inbounds x[i] = x[i] > threshold ? 1.0 : 0.0
  end
end
function binarize!(x::Vec{Float64};threshold=0.0)
  @simd for i=1:length(x)
    @inbounds x[i] = x[i] > threshold ? 1.0 : 0.0
  end
end

"""
  # Boltzmann.logsig (utils.jl)
  ## Function Calls
    `logsig(x::Mat{Float64})`
    `logsig(x::Vec{Float64})`
  
  ## Description
    Perform the logistic sigmoid element-by-element for the entries
    of `x`. Here, the logistic sigmoid is defined as
                      `y  =  1 / (1+e^-x)`.
    
  ## Returns
    1. `::Mat{Float64}` *or* `::Vec{Float64}` (depending on input).

  ### See also...
    `logsig!`
"""
function logsig(x::Mat{Float64})
    @devec s = 1 ./ (1 + exp(-x))
    return s
end
function logsig(x::Vec{Float64})
    @devec s = 1 ./ (1 + exp(-x))
    return s
end


"""
  # Boltzmann.logsig! (utils.jl)
  ## Function Calls
    `logsig!(x::Mat{Float64})`
    `logsig!(x::Vec{Float64})`
  
  ## Description
    Perform the logistic sigmoid element-by-element for the entries
    of `x`. Here, the logistic sigmoid is defined as
                      `y  =  1 / (1+e^-x)`.
    
  ## Returns
    Nothing. Modifies `x` in place.

  ### See also...
    `logsig`
"""
function logsig!(x::Mat{Float64})
    @devec x = 1 ./ (1 + exp(-x))
end
function logsig!(x::Vec{Float64})
    @devec x = 1 ./ (1 + exp(-x))
end

