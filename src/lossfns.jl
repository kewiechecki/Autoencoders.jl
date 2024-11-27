#[Float] -> Float
#Shannon entroy
function entropy(W::AbstractArray)
    W = abs.(W) .+ eps(eltype(W))
    W = W ./ sum(W)
    return -sum(W .* log2.(W))
end

# [[Float]] -> [Float]
#row/coumn Shannon entropy (default := column)
function entropy(W::AbstractMatrix;dims=1)
    W = abs.(W) .+ eps(eltype(W))
    W = W ./ sum(W,dims=dims)
    return -sum(W .* log2.(W),dims=dims)
end

function L1(x::AbstractArray)
    return sum(abs.(x))
end
    
function L1_scaleinv(x::AbstractArray)
    x = abs.(x)
    x = invsum(x) * x' * invsum(x,2)
    return only(x)
end

function L1_cos(x::AbstractArray)
    x = (abs.(cossim(x')) * (invsum(x) * x')')' * invsum(x,2)
    return sum(x)
end

function L1_normcos(x::AbstractArray)
    x = (abs.(cossim(x')) * x * abs.(cossim(x)) * invsum(x)')' * invsum(x,2)
    return sum(x)
end

function L1(M::SparseEncoder,α,x)
    c = diffuse(M,x)
    return α * sum(abs.(c))
end

function L1_scaleinv(M::SparseEncoder,α,x)
    c = diffuse(M,x)
    return α * L1_scaleinv(c)
end

function L1_normcos(M::SparseEncoder,α,x)
    c = diffuse(M,x)
    return α * L1_normcos(c)
end

function L2(M::SparseEncoder,lossfn,x,y)
    return lossfn(M(x),y)
end

function loss(lossfn::Function)
    return (M,x,y)->lossfn(M(x),y)
end

function loss(M::SparseEncoder,α::AbstractFloat,lossfn::Function,
              x::AbstractArray,y::AbstractArray)
    # x = cu(x)
    # y = cu(y)
    return L1(M,α,x) + L2(M,lossfn,x,y)
end

# if α isn't specified just calculate L2
function loss(M::SparseEncoder,lossfn::Function,
              x::AbstractArray,y::AbstractArray)
    # x = cu(x)
    # y = cu(y)
    return L2(M,lossfn,x,y)
end

function loss_SAE(α::AbstractFloat,lossfn::Function,
                  x::AbstractArray,y::AbstractArray)
    return M->loss(M,α,lossfn,x,y)
end

function loss_SAE(M_outer::SparseEncoder,α::AbstractFloat,lossfn::Function,
                  x::AbstractArray)
    # x = cu(x)
    yhat = M_outer(x)
    f = M->L1(M,α,x) + L2(M,lossfn,x,yhat)
    return m->lossfn((M_outer[2] ∘ m ∘ M_outer[1])(x),yhat)
end

function loss_SAE(lossfn::Function,x::AbstractArray,y::AbstractArray)
    return M->loss(M,lossfn,x,y)
end

function loss_SAE(M_outer::SparseEncoder,lossfn::Function,x::AbstractArray)
    # x = cu(x)
    yhat = M_outer(x)
    f = M->L1(M,x) + L2(M,lossfn,x,yhat)
    return m->lossfn((M_outer[2] ∘ m ∘ M_outer[1])(x),yhat)
end

function loss_SAE(M_outer::SparseEncoder,α::AbstractFloat,lossfn::Function)
    return (M,x,_)->begin
        yhat = M_outer(x)
        E = diffuse(M_outer,x)
        l1 = L1(M,α,E)
        l2 = lossfn(decode(M_outer,M(E)),yhat)
        return l1 + l2,l1,l2
    end
end

function bic(k::Integer,n::Integer,L::AbstractFloat)
    return k*log(n) - 2*log(L)
end

function aic(k::Integer,L::AbstractFloat)
    return 2*k - 2*log(L)
end
