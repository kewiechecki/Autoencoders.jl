@doc raw"""
`EncoderBlock`

Parallelization for encoders.

See also: `Flux.Parallel`.
"""
struct EncoderBlock <: AbstractEncoder
    heads::Parallel
    decoder
end
@functor EncoderBlock

@doc raw"""
`EncoderBlock(f::(_ -> AbstractEncoder), h::Integer, connection, args...; kwargs...) -> EncoderBlock`

Constructor for `EncoderBlock` of size `h`. Applies `f(args...;kwargs...)` `h` times and wraps the result in `Flux.Parallel(connection,...)`.

See also: `Flux.Parallel`.
"""
function EncoderBlock(f::Union{Function,DataType},h::Integer,connection,args...;kwargs...)
    layers = Parallel(connection,map(_->f(args...;kwargs...),1:h)...)
    return EncoderBlock(layers)
end

@doc raw"""
`heads(M::EncoderBlock) -> M.heads.layers`

Accesor for layers contained in M.heads.
"""
function heads(M::EncoderBlock)
    return M.heads.layers
end

@doc raw"""
`size(M::EncoderBlock) -> Integer`

Returns the number of heads in `M`, which is `length(M.heads.layers`.
"""
function size(M::EncoderBlock)
    return length(M.heads.layers)
end

@doc raw"""
`conn(M::EncoderBlock) -> Function`

Accessor for `M.heads.connection`.

See also: `Flux.Parallel`
"""
function conn(M::EncoderBlock)
    return M.heads.connection
end

@doc raw"""
`map(f, M::EncoderBlock,args...,kwargs...) -> collection`

Applies `f(m, args...; kwargs...)` to each head in `M`.

See also: `Base.map`.
"""
function map(f::Function,M::EncoderBlock,args...;kwargs...)
    return map(m->f(m,args...;kwargs...),heads(M))
end

@doc raw"""
`mapreduce(f, g, M::EncoderBlock, args...; kwargs...) -> typeof(g(...))`
`mapreduce(f, M::EncoderBlock, args...; kwargs...) -> typeof(g(...))`

Calls `map(f, M, args...; kwargs...) and wraps the result in `g`. If `g` is not given, uses `conn(M)`.
See also: `map`, `conn`, `Base.mapreduce`.
"""
function mapreduce(f::Function,g::Function,M::EncoderBlock,
                   args...;kwargs...)
    return reduce(g,map(f,M,args...;kwargs...))
end

function mapreduce(f::Function,M::EncoderBlock,args...;kwargs...)
    return reduce(conn(M),map(f,M,args...;kwargs...))
end

function encode(M::EncoderBlock,x)
    return mapreduce(encode,vcat,M,x)
end

function diffuse(M::EncoderBlock,x)
    return encode(M,x)
end

function decode(M::EncoderBlock,x)
    return M.decoder(x)
end

function (M::EncoderBlock)(x)
    return decode(M,diffuse(M,x))
end

