using Autoencoders, TrainingIO

using MLDatasets,OneHotArrays

epochs = 100
batchsize=512

# learning rate
η = 0.001

# weight decay rate
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

# sparsity coefficient
α = 0.001

# load MNIST
dat = MNIST(split=:train)[:]
target = onehotbatch(dat.targets,0:9)

# split MNIST into batchsize of batchsize
loader = mnistloader(batchsize)

encoder = mnistenc()
decoder = mnistdec()
classifier = mnistclassifier()