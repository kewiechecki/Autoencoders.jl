module Autoencoders

import Base.map
import Base.size
import Base.mapreduce

using Flux, Functors, CUDA, Zygote, LinearAlgebra
using MLDatasets,OneHotArrays
using DocumenterCitations

export SparseEncoder,Autoencoder,SAE,
    encode,decode,diffuse

export EncoderBlock, heads,size,conn,map,mapreduce

export euclidean,inveucl,cossim,sindiff,maskI,invsum,zerodiag,wak,pwak
export DistEnc,kern,dist

export SparseClassifier,PSAE,SparseDict,
    cluster,centroid,partition,encodepred
export SparseDict, DictEnc, dict

export entropy,L1,L1_scaleinv,L1_cos,L1_normcos,L2,loss,loss_SAE
export aic,bic

export zeroabl,meanabl
export sampledat,scaledat,unhot

export mnistenc, mnistdec, mnistclassifier, mnistloader

@doc raw"""
`AbstractEncoder`

Supertype of all encoder models in this package. All subtypes should be callable and implement `fmap`, `encode`, `decode`, and `diffuse`.

See also: `AbstractParitioned`, `Autoencoder`.
"""
abstract type AbstractEncoder
end

function (M::AbstractEncoder)(x)
    return decode(M,diffuse(M,x))
end

include("SAE.jl")
include("EncoderBlock.jl")

include("distfns.jl")

@doc raw"""
`abstract type AbstractDDAE <: AbstractEncoder`

Class of `Autoencoder` where reconstruction is performed using a diffusion kernel. In addition to methods inherited from `AbstractEncoder`, all subtypes should implement `kern`.

See also: `wak`, `pwak`, `DistEnc`, `AbstractParitioned`.
"""
abstract type AbstractDDAE
end

function encode(M::AbstractDDAE,X)
    return encode(M.autoencoder,X)
end

function decode(M::AbstractDDAE,X)
    return decode(M.autoencoder,X)
end

function diffuse(M::AbstractDDAE,X::AbstractArray{<:AbstractFloat})
    E = encode(M,X)
    D = kern(M,E)
    return (D * E')'
end

include("DistEnc.jl")

@doc raw"""
`AbstractPartitioned <: AbstractDDAE`

Subtypes of `AbstractEncoder` that attempt to find latent classifications in a dataset.

See also: `PAE`, `DistEnc`, `AbstractDDAE`, `AbstractEncoder`.
"""
abstract type AbstractPartitioned <: AbstractEncoder
end

@doc raw"""
`cluster(M:: <: AbstractPartitioned, X) -> typeof(X)`


See also: `centroid`, `diffuse`, `kern`, `partition`, `AbstractPartitioned`.
"""
function cluster(M::AbstractPartitioned,X)
    return (softmax ∘ M.partitioner)(X)
    #return (M.partitioner)(X)
end

@doc raw"""
`centroid(M:: <: AbstractPartitioned, X) -> typeof(X)`


See also: `cluster`, `diffuse`, `kern`, `partition`, `encodepred`, `AbstractPartitioned`.
"""
function centroid(M::AbstractPartitioned,X)
    E = encode(M,X)
    K = cluster(M,X)
    Ksum = sum(K,dims=2)

    C = K * E' ./ Ksum
    return C'
end

@doc raw"""
`partition(M:: <: AbstractPartitioned, X) -> typeof(X)`


See also: `cluster`, `diffuse`, `kern`, `centroid`, `encodepred`, `AbstractPartitioned`.
"""
function partition(M::AbstractPartitioned,X)
    return (pwak ∘ cluster)(M,X)
end

include("PartitionedEncoders.jl")
include("DistPart.jl")

include("SparseDict.jl")
include("DictEnc.jl")

include("lossfns.jl")
include("ablations.jl")

include("mnist.jl")
include("preprocessing.jl")
end # module Autoencoders
