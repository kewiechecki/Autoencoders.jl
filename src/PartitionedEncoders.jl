@doc raw"""
`PAE(autoencoder <: AbstractEncoder, partitioner) <: AbstractPartitioned`

Partitioned autoencoder.

See also: `encode`, `decode`, `diffuse`, `cluster`, `centroid`, `DistPart`.
"""
struct PAE <: AbstractPartitioned
    autoencoder::AbstractEncoder
    partitioner
end
@functor PAE

@doc raw"""
`encodepred(M:: <: AbstractPartitioned, X) -> typeof(X)`


See also: `cluster`, `diffuse`, `kern`, `centroid`, `partition`, `AbstractPartitioned`.
"""
function encodepred(M::PAE,X)
    E = encode(M,X)
    P = partition(M,X)
    return (P * E')'
end

function diffuse(M::PAE,X)
    return encodepred(M,X)
end

function (M::PAE)(X::AbstractMatrix)
    Ehat = encodepred(M,X)
    return decode(M,Ehat)
end
