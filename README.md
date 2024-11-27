# Autoencoders.jl

Autoencoder architectures for data analysis and mechanistic interpretability.

## Installation

```{julia}
] add https://github.com/kewiechecki/Autoencoders.jl
```

## Usage

```julia
using Autoencoders

using MLDatasets, StatsPlots, OneHotArrays
```
Training hyperparameters
```{julia}
epochs = 100
batchsize=512

# learning rate
η = 0.001

# weight decay rate
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

# sparsity coefficient
α = 0.001
```
Load data
```{julia}
dat = MNIST(split=:train)[:]
target = onehotbatch(dat.targets,0:9)

m_x,m_y,n = size(dat.features)
X = reshape(dat.features[:, :, :], m_x, m_y, 1, n)

loader = Flux.DataLoader((X,target),
                         batchsize=batchsize,
                         shuffle=true)
```