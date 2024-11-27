
@doc raw"""
`DistPart(autoencoder <: AbstractEncoder, partitioner, metric) <: AbstractPartitioned`



See also: `encode`, `decode`, `diffuse`, `cluster`, `centroid`, `PAE`, `DistEnc`.
"""
struct DistPart <: AbstractPartitioned
    autoencoder::AbstractEncoder
    partitioner::Chain
    metric::Function
end
@functor DistPart

function kern(M::DistPart,X::AbstractArray{<:AbstractFloat})
    E = encode(M,X)
    D = dist(M,E)
    P = partition(M,X)
    return wak(P .* D)
end

function diffuse(M::DistPart,X::AbstractArray{<:AbstractFloat})
    E = encode(M,X)
    D = dist(M,E)
    P = partition(M,X)
    G = wak(P .* D)
    return (G * E')'
end
