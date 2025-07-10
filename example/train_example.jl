using imageGenerationWithDiffusionModels
using Statistics
import imageGenerationWithDiffusionModels.ReverseSampling

using FileIO, PNGFiles
using Images
using BSON: @load

# Training

num_timesteps = 500 # for recreating alpha bar in terms of reverese sampling

training = false

if training
    model = train(num_timesteps=num_timesteps)
else
    @load "./example/model.bson" model
end

# Reverse Sampling

beta = cosine_beta_schedule(num_timesteps)
alphaBar = cumprod(1 .- beta)

#sample one greyscale image with the UNet
img = ReverseSampling.reverse_sample(model,
                                   (32,32,1,1);     
                                   T = num_timesteps,
                                   alpha_hats = alphaBar)

img = reshape(img, 32, 32) # remove additional batch dimensions
img = (img .- minimum(img)) ./ (maximum(img) - minimum(img)) # now in [0,1]
img = RGB.(img, img, img) # 32×32 Array{RGB}

save("./example/reverse_sample.png", img)
println("Image generated and saved!")
