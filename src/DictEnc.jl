@doc raw"""
`DistPart(autoencoder <: AbstractEncoder, partitioner, metric) <: AbstractPartitioned`



See also: `encode`, `decode`, `diffuse`, `cluster`, `centroid`, `PAE`, `DistEnc`.
"""
struct DictEnc <: AbstractPartitioned
    dict::SparseDict
    classifier
    decoder
end
@functor DictEnc

function DictEnc(classifier::Chain,decoder::Chain,d::Integer,k::Integer)
    dict = SparseDict(d,k)
    return DictEnc(dict,classifier,decoder)
end

function DictEnc(m::Integer,d::Integer,k::Integer,σ=relu)
    dict = SparseDict(d,k)
    classifier = Chain(Dense(m => k,σ))
    decoder = Chain(Dense(d => m,σ))
    return DictEnc(dict,classifier,decoder)
end

function cluster(M::DictEnc,X)
    return (softmax ∘ M.classifier)(X)
end

function dict(M::DictEnc)
    return M.dict.dict
end

function encode(M::DictEnc,X)
    return diffuse(M,X)
end

function diffuse(M::DictEnc,X)
    K = cluster(M,X)
    return M.dict(K)
end

function centroid(M::DictEnc,X)
    return M.dict.dict
end

function decode(M::DictEnc,E)
    return M.decoder(E)
end

function (M::DictEnc)(X)
    K = cluster(M,X)
    E = M.dict(K)
    return decode(M,E)
end
