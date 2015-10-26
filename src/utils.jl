

if !isdefined(:__EXPRESSION_HASHES__)
    __EXPRESSION_HASHES__ = Set{Uint64}()
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

typealias Mat{T} AbstractArray{T, 2}
typealias Vec{T} AbstractArray{T, 1}

function normalize_samples(X)
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

function normalize_samples!(X)
    samples = size(X,2)

    for i=1:samples
      x = X[:,i]
      minx = minimum(x)
      maxx = maximum(x)
      ranx = maxx-minx
      X[:,i] = (x-minx)/ranx
    end
end

function normalize!(x)
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds x[i] = (x[i]-minx) / ranx
    end
end

function normalize(x)
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds x[i] = (x[i]-minx) / ranx
    end

    return x
end

function binarize!(x;level=0.001)
  @simd for i=1:length(x)
    @inbounds x[i] = x[i] > level ? 1.0 : 0.0
  end
end

function binarize(x;level=0.001)
  @simd for i=1:length(x)
    @inbounds x[i] = x[i] > level ? 1.0 : 0.0
  end
  return x
end