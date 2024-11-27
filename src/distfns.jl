@doc raw"""
`clusts(C <: AbstractMatrix) -> AbstractArray{Integer}`

Converts soft classifications into numeric labels. `C` should be a matrix where columns are samples and rows give scores with respect to classifications.
"""
function clusts(C::AbstractMatrix)
    return map(x->x[1],argmax(C,dims=1))
end

@doc raw"""
`neighborcutoff(G <: AbstractArray; ϵ=0.0001) -> typeof(G)`

Masks all cells in `G .<= ϵ`.
""" 
function neighborcutoff(G::AbstractArray; ϵ=0.0001)
    M = G .> ϵ
    return G .* M
end

@doc raw"""
`euclidean(x <: AbstractArray; dims=1) -> typeof(x)`

Version of Euclidean distance compatible with Flux's automatic differentiation. Calculates pairwise distance matrix by column (default) or row

See also: `inveucl`,`CuArray`, `Distances.Euclidean`.
"""
#function euclidean(x::CuArray{Float32};dims=1)
function euclidean(x::AbstractArray{<:AbstractFloat};dims=1)
    x2 = sum(x .^ 2, dims=dims)
    D = x2' .+ x2 .- 2 * x' * x
    # Numerical stability: possible small negative numbers due to precision errors
    D = sqrt.(max.(D, 0) .+ eps(Float32))  # Ensure no negative values due to numerical errors
    return D
end

@doc raw"""
inveucl(x <: AbstractArray] -> typeof(x)

Returns reciprocal Euclidean distance matrix. The intended use is to construct adjacency matrices where edge weight is directly proportional to proximity.

See also: `euclidean`
"""
function inveucl(x::AbstractArray;dims=1)
    return 1 ./ (euclidean(x) .+ eps(Float32))
end

@doc raw"""
`cossim(x <: AbstractArray{<:AbstractFloat}; dims = 1) -> typeof(x)`

Function to calculate cosine similarity matrix.

See also: `sindiff`.
"""
function cossim(x::AbstractArray{<:AbstractFloat};dims=1)
    # Normalize each column (or row, depending on 'dims') to unit length
    norms = sqrt.(sum(x .^ 2, dims=dims) .+ eps(Float32))
    x_normalized = x ./ norms

    # Compute the cosine similarity matrix
    # For cosine similarity, the matrix multiplication of normalized vectors gives the cosine of angles between vectors
    C = x_normalized' * x_normalized

    # Ensure the diagonal elements are 1 (numerical stability)
    #i = CartesianIndex.(1:size(C, 1), 1:size(C, 1))
    #C[i] .= 1.0

    return C
end

@doc raw"""
`sindiff(x <: AbstractArray{<:AbstractFloat}; dims=1) -> typeof(x)`


See also: `cossim`.
"""
function sindiff(x::AbstractArray{<:AbstractFloat};dims=1)
    C = cossim(x;dims=dims)
    return sqrt.(max.(1 .- C .^ 2,0))
end

@doc raw"""
`maskI(n::Integer) -> CuArray{Int64,2}`

Returns an `n × n` `CuArray` with all entries in the diagonal set to 0 and all other entries set to 1. The intended usage is to mask the diagonal of a `CuArray`. 

See also: `LinearAlgebra.I`, `LinearAlgebra.diag`,`CUDA.CuArray`
"""
function maskI(n::Integer)
    return 1 .- I(n) |> gpu
end

@doc raw"""
`invsum(x <: AbstractArray; dims=1) -> typeof(x)`

Returns `1 ./ sum(x; dims)`. A constant `eps()` is added to avoid divide-by-zero errors.

See also: `eps`, `sum`.
"""
function invsum(x::AbstractArray; dims=1)
    return 1 ./ sum(x .+ eps(),dims=dims)
end

@doc raw"""
`zerodiag(G <: AbstractArray) -> typeof(G)`
`zerodiag(G <: CuArray) -> typeof(G)`

Returns `G` with diagonal entries set to 0. It is only defined for square matrices.

See also: `maskI`, `LinearAlgebra.I`
"""
function zerodiag(G::AbstractArray)
    m, n = size(G)
    G = G .* (1 .- I(n))
    return G
end

# [CuArray] -> [CuArray]
# workaround for moving identity matrix to GPU 
function zerodiag(G::CuArray)
    m, n = size(G)
    G = G .* maskI(n) #(1 .- I(n) |> gpu)
    return G
end

@doc raw"""
`wak(G <: AbstractArray; dims=1) -> typeof(G)`

Constructs a weighted affinity kernel from an adjacency matrix. The diagonal is set to 0 and values are normalized such that `sum(G; dims=dims) .== 1`. It is only defined for square matrices. If `dims=1`, `wak(G)` will be a right stochastic matrix. If `dims=2`, `wak(G)` will be a left stochastic matrix.

This transformation allows `wak(G)` to be used as a denoising diffusion kernel[Tjarnberg2021](@cite). Effectively, denoising is treated as a Markov process. Each value in a data set (pixels in an image, genes in a sequencing experiment &c.) is interpolated by taking a weighted average of some subset of the data (If `G` is interpreted as an adjacency matrix, points are interpolated as a weighted average of their neighbors.). Importantly, points cannot be estimators for themselves[batson2019](@cite). If `wak(G)` is interpreted as a transition matrix, points cannot stay still.

For matrices `X` and `G` where `size(X) .== size(G)`, `(wak(G) * X')'` gives a denoised estimate `X̂` where each element is replaced by a weighted average of its neighbors.

If `sum(G[i,:]) == 0`, then `X̂[:,i] == 0`. If this is undesired behavior, you may instead use `wak(G .+ eps())`. This results 

See also: `pwak`, `zerodiag`
"""
function wak(G::AbstractArray; dims=1)
    G = zerodiag(G)
    G = G ./ (sum(G,dims=dims) .+ eps(eltype(G)))
    return G
end

@rdoc raw"""
`pwak(K <: AbstractMatrix; dims=1) -> typeof(K)`

Constructs a "partitioned weighted affinity kernel". Like `wak`, but instead of accepting an adjacency matrix, it accepts an `m × n` matrix `K` where each entry `K[i,j]` is the proability that sample `j` is in category `i` and all categories are mutually exclusive.

Like `wak(G)` `pwak(K)` can be used as a transition matrix. The transition probability from one sample to another is equivalent to the probability that the samples are in the same category.

`pwak(K)` can be used to prune edges from an adjacency matrix between nodes in different categories (or "soft prune" edges between nodes that are unlikely to fall in the same category) with `Ĝ = G .* pwak(K)`.

This allows denoising diffusion to be used as a clustering algorithm.

See also: `wak`.
"""
function pwak(K::AbstractMatrix; dims=1)
    P = K' * K
    return wak(P)
end
