@doc raw"""
`Autoencoder(encoder::(T_e -> T_d), decoder::(T_d -> T_e)) -> T_e -> T_e

Simplest autoencoder type. Wraps `encoder` and `decoder` models and provides `encode` and `decode` methods. `encoder` and `decoder` should both have type `* <: AbstractArray -> * <: AbstractArray`. 

See also: `encode`, `decode`, `diffuse`, `AbstractEncoder`.
"""
struct Autoencoder <: AbstractEncoder
    encoder
    decoder
end
@functor Autoencoder

@doc raw"""
`encode(M:: <:AbstractEncoder, X) -> typeof(X)


See also: `diffuse`, `decode`, `Autoencoder`.
"""
function encode(M::Autoencoder,x)
    return M.encoder(x)
end

@doc raw"""
`diffuse(M:: <: AbstractEncoder, X) -> typeof(X)

If `typeof(M)` is a diffusion model (`<:AbstractDDAE`), applies diffusion to `encode(M,X)`. For `E = encode(M,X)`, this is usually given by `(kern(M,E) * E')'`. For other model types it is equivalent to `encode(M,X)`.

See also: `encode`, `decode`, `AbstractDDAE`, `wak`, `kern`.
"""
function diffuse(M::Autoencoder,X)
    return encode(M,X)
end

@doc raw"""
`decode(M:: <:AbstractEncoder, E) -> typeof(E)


See also: `encode`, `diffuse`, `Autoencoder`.
"""
function decode(M::Autoencoder,E)
    return M.decoder(E)
end


@doc raw"""
`SAE(weight,bias,σ)`

Canonical sparse autoencoder architecture adapted from [1]. Weights are symmetric. For nonsymmetric weights, define separate encoder and decoder layers using the `Autoencoder` class.

[1] Bricken, et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning", Transformer Circuits Thread, 2023.
"""
struct SAE <: AbstractEncoder
    weight::AbstractArray
    bias::AbstractArray
    σ::Function
end
# the magic line that makes everything "just work"
@functor SAE 

# constructor specifying i/o dimensions
function SAE(m,d,σ=relu)
    weight = randn(d,m)
    bias = randn(d)
    return SAE(weight,bias,σ)
end

#
function encode(M::SAE,x)
    return M.σ.(M.weight * x .+ M.bias)
end

function decode(M::SAE,c)
    return M.weight' * c
end

function diffuse(M::SAE,X)
    return encode(M,X)
end
