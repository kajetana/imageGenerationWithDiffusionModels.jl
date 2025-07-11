using imageGenerationWithDiffusionModels
using Statistics
import imageGenerationWithDiffusionModels.ReverseSampling

using FileIO, PNGFiles
using Images
using BSON: @load

num_timesteps = 100
training = false

if training
    model = train(num_timesteps=num_timesteps)
elseif isfile("example/model.bson")
    @load "example/model.bson" model

    # Reverse Sampling
    beta = cosine_beta_schedule(num_timesteps)
    alphaBar = cumprod(1 .- beta)

    img = ReverseSampling.reverse_sample(model,
                                         (32, 32, 1, 1);
                                         T = num_timesteps,
                                         alpha_hats = alphaBar)

    img = reshape(img, 32, 32)
    img = (img .- minimum(img)) ./ (maximum(img) - minimum(img))
    img = RGB.(img, img, img)

    save("reverse_sample.png", img)
    println("Image generated and saved!")
else
    println("Model file is not uploaded to github due to size restrictions. Please upload it from:")
    println("https://drive.google.com/drive/folders/1cL-ZlGzCGJ8lYINLyfVwRAx8p6VA8Ygy")
end

