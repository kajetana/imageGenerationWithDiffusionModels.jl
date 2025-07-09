# include("imageGenerationWithDiffusionModels.jl")
# using .imageGenerationWithDiffusionModels
# include("reverse_sampling.jl")

using Flux
using ImageView
using BSON: @save, @load

function train(;FILE_PATH = "./example/SyntheticImages500.mat", 
    learning_rate = 0.0001,
    epochs = 15,
    batch_size = 32,
    shuffle = true,
    model = unet(
        1,
        5,
        16,
        imageGenerationWithDiffusionModels.LearnedTEmbedding(128),
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
    data = reshape(data, 32, 32, 1, :)
    
    # noising variables
    num_timesteps = 100
    beta = imageGenerationWithDiffusionModels.cosine_beta_schedule(num_timesteps) # cosine schedule
    alphaBar = cumprod(1 .- beta)


    optimizer = Flux.setup(Adam(learning_rate), model)
    training_data = Flux.DataLoader((data, ), batchsize=batch_size, shuffle=shuffle)
    training = true
    ###############################################################################################################
    # training the model
    #
    # credits: 
    # https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#forward-diffusion
    # https://adrianhill.de/julia-ml-course/L7_Deep_Learning/
    # https://github.com/ytdeepia/DDPM/blob/main/src/training.py
    # https://docs.julialang.org/en/v1/stdlib/Random/
    ###############################################################################################################
    if training
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
                    println("  Batch $batch_idx, Loss: $(loss_val)")
                end
            end
        end

        println("Training finished!")

        # Save the model
        @save "model.bson" model
    else
        @load "model.bson" model
    end 
end

export train