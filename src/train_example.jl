include("imageGenerationWithDiffusionModels.jl")
using .imageGenerationWithDiffusionModels
include("reverse_sampling.jl")

using Flux
using ImageView
using BSON: @save, @load
using Statistics

###############################################################################################################
# loading and preprocessing data
#
# credits: 
# https://adrianhill.de/julia-ml-course/L7_Deep_Learning/
###############################################################################################################

const FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")

data = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)["syntheticImages"]

# "To make our data compatible with Flux models, we need to add a singleton 
# color-channel to x to make it compatible with convolutional layers"
data = reshape(data, 32, 32, 1, :)
#print(typeof(data))

###############################################################################################################
# creating the model
#
# credits: 
# https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#forward-diffusion
###############################################################################################################

# in_channels::Int,
# num_levels::Int,
# model_dim::Int,
# time_embed,
# emb_dim::Int;
# block_layer=TResBlock,
# num_blocks_per_level::Int=1

#in_channels = size(data, 3)

model = imageGenerationWithDiffusionModels.unet(
    1,
    4,
    16,
    imageGenerationWithDiffusionModels.LearnedTEmbedding(128),
    128;
    num_blocks_per_level=1)

###############################################################################################################
# training the model
#
# credits: 
# https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#forward-diffusion
# https://adrianhill.de/julia-ml-course/L7_Deep_Learning/
# https://github.com/ytdeepia/DDPM/blob/main/src/training.py
# https://docs.julialang.org/en/v1/stdlib/Random/
###############################################################################################################

# training variables
learning_rate = 0.001
epochs = 25
batch_size = 32
shuffle = true

# noising variables
num_timesteps = 100
#beta = imageGenerationWithDiffusionModels.cosine_beta_schedule(num_timesteps) # cosine schedule
beta =  LinRange(1e-4, 0.02, 100) # linear schedule
alphaBar = cumprod(1 .- beta)

# optimizer
optimizer = Flux.setup(Adam(learning_rate), model)

# training set, no classification, unsupervised training
training_data = Flux.DataLoader((data, ), batchsize=batch_size, shuffle=shuffle)

training = true

if training
    losses = Float32[]

    println("Training...")
    for epoch in 1:epochs

        losses_epoch = Float32[]

        for batch in training_data
            batch = batch[1]

            imgs = similar(batch)
            noise = similar(batch)

            timesteps = rand(1:num_timesteps, size(batch, 4))

            # iterate over the images in batch and apply noise
            for i in 1:size(batch, 4)
                imgs[:, :, :, i], noise[:, :, :, i] = imageGenerationWithDiffusionModels.add_noise_to_image(batch[:, :, :, i], timesteps[i], alphaBar)
            end

            loss, grads = Flux.withgradient(m -> Flux.mse(model(imgs, timesteps), noise), model)

            Flux.update!(optimizer, model, grads[1])

            push!(losses, loss)

            push!(losses_epoch, loss)

        end

        println("Mean Loss in Epoch: ", mean(losses_epoch))
    end

    println("Training finshed!")

    # credits for saving/loading the mode
    # https://stackoverflow.com/questions/68335891/how-to-load-a-trained-model-with-bson-in-flux-jl

    @save "model.bson" model
else
    @load "model.bson" model
end 

###############################################################################################################
# reverse sampling
##############################################################################################################

x = ReverseSampling.reverse_sample(model, (32, 32, 1, 1), T=100, alpha_hats=alphaBar)
#print(size(x))

x = reshape(x, 32, 32)

img = rand(32,32)
gui = ImageView.imshow(img)
canvas = gui["gui"]["canvas"]

ImageView.imshow(canvas, x)
sleep(8.0)

ImageView.close(gui["gui"]["window"])