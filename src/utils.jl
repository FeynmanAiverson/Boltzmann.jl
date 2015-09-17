

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

function ColumnNorms(x)
    return sqrt(sum(x.^2,1))
end

function UnitVarColumns(x)
    N = size(x,1)
    norms = sqrt(sum(x.^2,1))

    # x_i = x_i / ||x_i||
    x = broadcast(/,x, ColumnNorms(x)) * sqrt(Nu)   

    return x
end

function UnitVarColumns!(x)
    N = size(x,1)
    norms = sqrt(sum(x.^2,1))

    # x_i = x_i / ||x_i||
    x = broadcast(/,x, ColumnNorms(x)) * sqrt(Nu)   
end

function RemoveMean(x,dim)
    x = broadcast(-,x,mean(x,dim))
    return x
end

function RemoveMean!(x,dim)
    x = broadcast(-,x,mean(x,dim))
end