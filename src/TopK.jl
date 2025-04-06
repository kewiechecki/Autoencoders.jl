struct TopK
    k
    dims
    reverse
end

function TopK(k; dims=1, reverse=false)
    TopK(k,dims,reverse)
end

@doc raw"""
`maskmax(X <: AbstractArray; dims=1, reverse=false) -> typeof(X)`

Returns 1-hot mask for max values of `X` along `dims`.
    """
function maskmax(X::CuMatrix; dims=1, reverse=false)
    m,n = size(X)
    ij = argmax(X; dims=dims)
    IJ = map(d->(i->i.I[d]).(ij),1:2)
    I,J = reshape.(IJ,:)
    V = CuArray(rep(true,length(ij)))
    sparse(I,J,V,m,n)
end

    
@doc raw"""
`topk(X <: AbstractArray, k::Integer; dims=1, reverse=false) -> typeof(X)`

Generates mask for top `k` values in `X`.
    """
function topk(X, k; dims=1, reverse=false)
    m,n = size(X)
    f = (M,_) -> M .| maskmax(X .* (.! M); dims=dims, reverse=reverse)
    init = CUDA.zeros(Bool,m,n)
    foldl(f, 1:k; init=init)
end

function(l::TopK)(X)
    G = topk(X, l.k; dims=l.dims, reverse=l.reverse)
    X .* G
end


function Zygote.pullback(l::TopK, X::AbstractArray)
    G = topk(X, l.k; dims=l.dims, reverse=l.reverse)
    Y = X .* G  # Forward pass

    function B(Δ)
        # Gradient w.r.t. X
        ∇_X = G .* Δ
        # Return the gradients in the expected structure by Zygote
        return (Nothing,), ∇_X
    end
    
    return E, B
end
