module Training

using ..imageGenerationWithDiffusionModels
using ..Embeddings
using ..Blocks
using ..FeatureEncoderNetwork
using ..UNet
using ..Scheduler
using ..ReverseSampling

using Flux
using BSON: @save

"""
    train(;FILE_PATH::String = "./example/SyntheticImages500.mat",
    num_timesteps::Int = 100,
    learning_rate::Real = 0.0001,
    epochs::Int = 15,
    batch_size::Int = 32,
    shuffle::Bool = true,
    model::unet = unet(
        1,
        5,
        16,
        LearnedTEmbedding(128),
        128;
        num_blocks_per_level=1
    ))

Trains a diffusion model on a digit dataset.

# Arguments
- `FILE_PATH::String` : A filepath to .mat digits data. Defaults to `"./example/SyntheticImages500.mat"`.
- `num_timesteps::Int` : Number of noising steps. Defaults to `100`.
- `cosine::Bool` : Cosine beta scheduling. Otherwise linear. Defaults to `true`.
- `learning_rate::Real` : A training learning rate. Defaults to `0.0001`.
- `epochs::Int` : Number of training epochs. Defaults to `15`.
- `batch_size::Int` : Size of training batches. Defaults to `32`.
- `shuffle::Bool` : Shuffle training data. Defaults to `true`.
- `model::unet` : Diffusion model. Defaults to `unet(1, 5, 16, LearnedTEmbedding(128), 128; num_blocks_per_level=1)`.

# Returns
A trained diffusion model. Saves the model as "model.bson".
"""
function train(;FILE_PATH::String = "./example/SyntheticImages500.mat",
    num_timesteps::Int = 100,
    cosine::Bool = true,
    learning_rate::Real = 0.0001,
    epochs::Int = 15,
    batch_size::Int = 32,
    shuffle::Bool = true,
    model::unet = unet(
        1,
        5,
        16,
        LearnedTEmbedding(128),
        128;
        num_blocks_per_level=1
    ))
    
    ###############################################################################################################
    # loading and preprocessing data
    #
    # credits: 
    # https://adrianhill.de/julia-ml-course/L7_Deep_Learning/
    ###############################################################################################################
    
    data = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)["syntheticImages"]

    # "To make our data compatible with Flux models, we need to add a singleton 
    # color-channel to x to make it compatible with convolutional layers"   
    data = reshape(data, 32, 32, 1, :)
    
    # noising variables
    if cosine
        beta = cosine_beta_schedule(num_timesteps) # cosine schedule
    else
        beta = LinRange(1e-4, 0.02, num_timesteps) # linear schedule
    end
    
    alphaBar = cumprod(1 .- beta)

    optimizer = Flux.setup(Adam(learning_rate), model)
    training_data = Flux.DataLoader((data, ), batchsize=batch_size, shuffle=shuffle)

    ###############################################################################################################
    # training the model
    #
    # credits: 
    # https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#forward-diffusion
    # https://adrianhill.de/julia-ml-course/L7_Deep_Learning/
    # https://github.com/ytdeepia/DDPM/blob/main/src/training.py
    # https://docs.julialang.org/en/v1/stdlib/Random/
    ###############################################################################################################

    losses = Float32[]

    println("Training...")
    for epoch in 1:epochs
        println("Epoch $epoch/$epochs")
        epoch_losses = Float32[]
        
        for (batch_idx, batch) in enumerate(training_data)
            batch = batch[1]
            imgs = similar(batch)
            noise = similar(batch)

            timesteps = rand(1:num_timesteps, size(batch, 4))

            # iterate over the images in batch and apply noise
            for i in 1:size(batch, 4)
                imgs[:, :, :, i], noise[:, :, :, i] = imageGenerationWithDiffusionModels.add_noise_to_image(batch[:, :, :, i], timesteps[i], alphaBar)
            end
            
            # Normalize timesteps to [0, 1]
            timesteps_cont = (timesteps .- 1) ./ (num_timesteps - 1)
            
            # Define loss function
            function loss_fn(m)
                predicted_noise = m(imgs, timesteps_cont)
                return Flux.mse(predicted_noise, noise)
            end
            
            # Compute gradients
            loss_val, grads = Flux.withgradient(loss_fn, model)
            
            # Update model parameters
            Flux.update!(optimizer, model, grads[1])
            
            # Track loss
            push!(epoch_losses, loss_val)
            push!(losses, loss_val)
            
            if batch_idx % 10 == 0
                println("Batch $batch_idx, Loss: $(loss_val)")
            end
        end
    end

    println("Training finished!")

    # Save the model
    @save joinpath(@__DIR__, "", "model.bson") model

    return model
end

export train

end