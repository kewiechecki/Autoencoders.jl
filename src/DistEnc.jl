@doc raw"""
`DistEnc <: AbstractDDAE`

Class of `Autoencoder` that uses a distance metric to generate a diffusion kernel.
"""
struct DistEnc <: AbstractEncoder
    #encoder::Chain
    #decoder::Chain
    autoencoder::AbstractEncoder
    metric
end
@functor DistEnc

function DistEnc(m::Integer,d::Integer,σ::Function,metric::Function)
    θ = Chain(Dense(m => d,σ))
    ϕ = Chain(Dense(d => m,σ))
    return DistEnc(Autoencoder(θ,ϕ),metric)
end

function dist(M::DistEnc,E)
    return M.metric(E)
end

function kern(M::DistEnc,E)
    D = dist(M,E)
    return wak(D)
end

