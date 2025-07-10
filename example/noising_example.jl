using imageGenerationWithDiffusionModels
using Images

###############################################################################################################
# noising variables
###############################################################################################################

num_timesteps = 500

cosine = false

if cosine
    beta = cosine_beta_schedule(num_timesteps) # cosine schedule
else
    beta = LinRange(1e-4, 0.02, num_timesteps) # linear schedule
end

alphaBar = cumprod(1 .-beta)

ts = num_timesteps:-50:0 # noising steps

###############################################################################################################
# displaying the noising process
###############################################################################################################

const FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")
data = load_digits_data(FILE_PATH)

images = data["syntheticImages"]

# display the noising process for the first few images of the dataset
for i in 1:3
    img = images[:, :, 1, i]

    img = visualize_noising_of_image(img, ts, alphaBar)

    img = (img .- minimum(img)) ./ (maximum(img) - minimum(img)) # now in [0,1]
    img = RGB.(img, img, img) # 32×32 Array{RGB}

    save("noising" * string(i) * ".png", img)
end

println("Images generated!")
