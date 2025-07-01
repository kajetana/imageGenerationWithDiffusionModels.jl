include("imageGenerationWithDiffusionModels.jl")
using .imageGenerationWithDiffusionModels

using Flux

###############################################################################################################
# loading and preprocessing data
#
# credits: 
# https://adrianhill.de/julia-ml-course/L7_Deep_Learning/
###############################################################################################################

const FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")

data = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)["syntheticImages"]

data = reshape(data, 32, 32, 1, :)

###############################################################################################################
# creating the model
#
# credits: 
# https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#forward-diffusion
###############################################################################################################

model = imageGenerationWithDiffusionModels.unet(
    1,
    4,
    16,
    imageGenerationWithDiffusionModels.LearnedTEmbedding(128),
    128;
    num_blocks_per_level=1)

###

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
epochs = 10
batch_size = 32
shuffle = true

# noising variables
num_timesteps = 100
beta = imageGenerationWithDiffusionModels.cosine_beta_schedule(num_timesteps)
alphaBar = cumprod(1 .- beta)

# optimizer
optimizer = Flux.setup(Adam(learning_rate), model)

# training set, no classification, unsupervised training
training_data = Flux.DataLoader((data, ), batchsize=batch_size, shuffle=shuffle)

# test
#batch = first(training_data)
#println(size(batch[1]))
#println(size(batch[1][:, :, :, 1]))
#print(typeof(rand(1:num_timesteps, batch_size)))

losses = Float32[]

for epoch in 1:epochs
    for batch in training_data
        batch = batch[1]

        imgs = similar(batch)
        noise = similar(batch)

        timesteps = rand(1:num_timesteps, batch_size)

        # iterate over the images in batch and apply noise
        for i in 1:batch_size
            imgs[:, :, :, i], noise[:, :, :, i] = imageGenerationWithDiffusionModels.add_noise_to_image(batch[:, :, :, i], timesteps[i], alphaBar)
        end

        loss, grads = Flux.withgradient(m -> Flux.mse(model(imgs, timesteps), noise), model)

        Flux.update!(optimizer, model, grads[1])

        push!(losses, loss)
    end
end