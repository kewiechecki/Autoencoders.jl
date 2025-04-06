@doc raw"""
`mlp(l::AbstractVector{<: Integer},f::Function)`

Builds a dense MLP from a vector of layer widths `l`,
each with activation function `f`.

# Example
```julia-repl
julia> layers = [114, 58, 29, 14]
4-element Vector{Int64}:
 114
  58
  29
  14

julia> encoder = mlp(layers, tanh) |> gpu
Chain(
  Dense(114 => 58, tanh),               # 6_670 parameters
  Dense(58 => 29, tanh),                # 1_711 parameters
  Dense(29 => 14, tanh),                # 420 parameters
)                   # Total: 6 arrays, 8_801 parameters, 976 bytes.

julia> decoder = mlp(reverse(layers), tanh) |> gpu
Chain(
  Dense(14 => 29, tanh),                # 435 parameters
  Dense(29 => 58, tanh),                # 1_740 parameters
  Dense(58 => 114, tanh),               # 6_726 parameters
)                   # Total: 6 arrays, 8_901 parameters, 976 bytes.

julia> autoenc = Autoencoder(encoder,decoder)
Autoencoder(Chain(Dense(114 => 58, tanh), Dense(58 => 29, tanh), Dense(29 => 14, tanh)), Chain(Dense(14 => 29, tanh), Dense(29 => 58, tanh), Dense(58 => 114, tanh)))
```

See also: Flux.Dense, Flux.Chain
"""
function mlp(l::AbstractVector{<: Integer},f::Function)
    θ = foldl(l[3:length(l)],
              init=Chain(Dense(l[1] => l[2],f))) do layers,d
        d_0 = size(layers[length(layers)].weight)[1]
        return Chain(layers...,Dense(d_0 => d,f))
    end
end

