

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

function normalize(x)
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds x[i] = (x[i]-minx) / ranx
    end
    return x
end

function normalize!(x)
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds x[i] = (x[i]-minx) / ranx
    end
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