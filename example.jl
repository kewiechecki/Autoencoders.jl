using Autoencoders, TrainingIO, Flux, JLD2

using MLDatasets,OneHotArrays

# allows MNIST images to be viewed in terminal
using ImageInTerminal, Colors

function showdigit(img::Array{<:AbstractFloat,3})
    x,y,z = size(img)
    img = reshape(img,x,y)
    Gray.(img')
end

# where to save models
path = "tmp/" 

###############################
# define hyperparameters

epochs = 100
batchsize=512

# learning rate
η = 0.001

# weight decay rate
λ = 0.001

# sparsity coefficient
α = 0.001

optimiser = OptimiserChain(Flux.AdamW(η),Flux.WeightDecay(λ))
################################


################################
# load MNIST
# split into batchsizes of batchsize
# each element of loader is a pair of (image, label)
loader = mnistloader(batchsize)
################################


################################
# example submodels
encoder = mnistenc() |> gpu
decoder = mnistdec() |> gpu
classifier = mnistclassifier() |> gpu

encodeclassify = Chain(encoder,classifier)
################################


################################
# loss function
# loss(f) converts a binary loss function f::(X -> Y -> AbstractFloat) to a function g::((X -> Y) -> X -> Y -> AbstractFloat

# following mathematical conventions, "_" is used in these examples to denote subscripts
# thus "loss_classifier" is pronounced "loss with respect to classifier" rather than "loss classifier"
loss_classifier = loss(Flux.crossentropy)
################################


################################
L_classifier = train!(encodeclassify,
                      path*"/classifier",
                      loss_classifier,
                      loader,
                      optimiser,
                      epochs,
                      savecheckpts=false) # if true, saves a separate model each epoch

autoenc = Autoencoder(encoder,decoder)

loss_autoenc = loss(Flux.mse)
################################


################################
# train outer model
L = train!(autoenc,
           path*"/autoencoder",
           loss_autoenc,
           loader,
           optimiser,
           epochs,
           ignoreY = true, # autoenc tries to reconstruct the input rather than classifying it, so the loss function should compare loss against the input rather than the labels
           savecheckpts=false)
################################


################################
# compare reconstructed output

# examine first minibatch 
x,y = first(loader)
showdigit(x[:,:,:,1])

x̂ = autoenc(x |> gpu) |> cpu
showdigit(x̂[:,:,:,1])
################################


################################
# train multimodal model
# paired encoder-classifier

# return tuple of decoder, classifier output
combine = (args...)->tuple(args...)
multimodal = Autoencoder(encoder,Parallel(combine,decoder,classifier))

loss_multimodal = (M,x,y)->begin
    x̂,ŷ = M(x)
    L_decoder = Flux.mse(x̂,x)
    L_classifier = Flux.crossentropy(ŷ,y)
    return L_decoder + L_classifier, L_decoder, L_classifier
end

L = train!(multimodal,
           path*"/multimodal",
           loss_multimodal,
           loader,
           optimiser,
           epochs,
           savecheckpts=false)
################################


################################
# load trained model
@load path*"/multimodal/final.jld2" state 
Flux.loadmodel!(multimodal,state)
################################


################################
# reconstructed inage, label
x̂,ŷ = multimodal(x |> gpu) |> cpu
showdigit(x̂[:,:,:,1])
################################


################################

# sparse autoencoder

sae = SAE(3,50,relu) |> gpu

# instead of training against object-level loss, we can train the SAE to reconstruct the embeddings
loss_SAE = (M,x,_)->begin
    E = diffuse(multimodal,x)
    F = diffuse(M,E)
    Ê = decode(M,F)

    x,y = decode(multimodal,E)
    x̂,ŷ = decode(multimodal,Ê)
    
    sparsity = L1(F)
    L_decoder = Flux.mse(x̂,x)
    L_classifier = Flux.crossentropy(ŷ,y)
    L = sparsity + L_decoder + L_classifier
    return L, sparsity, L_decoder, L_classifier
end

L_SAE = train!(sae,
           path*"/SAE",
           loss_SAE,
           loader,
           optimiser,
           epochs,
           savecheckpts=false)
################################


################################
# embeddings
E = encode(multimodal,x|>gpu)

# reconstructed output
x̂,ŷ = decode(multimodal,sae(E))
################################


################################
# other variations
sae_eucl = DistEnc(sae,inveucl)

partitioner = Chain(Dense(3,50,relu),Dense(50,10,relu)) |> gpu
sae_part = PAE(sae,partitioner)

sae_distpart = DistPart(sae,partitioner,inveucl)

dictdecoder = Chain(Dense(50 => 3,relu))
sae_dict = DictEnc(partitioner,dictdecoder,50,10) |> gpu
################################
