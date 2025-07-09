using imageGenerationWithDiffusionModels

###############################################################################################################
# noising variables
###############################################################################################################

const FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")
beta =  LinRange(1e-4, 0.02, 500)  # posterior variance
alphaBar = cumprod(1 .-beta)
ts = 500:-50:0 # noising steps
data = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)  # Explicitly reference the module 

###############################################################################################################
# displaying the noising process
###############################################################################################################

images = data["syntheticImages"]

# display the noising process for the first few images of the dataset
for i in 1:3
    img = images[:, :, 1, i]

    img = imageGenerationWithDiffusionModels.visualize_noising_of_image(img, ts, alphaBar)

    img = (img .- minimum(img)) ./ (maximum(img) - minimum(img)) 
    img = RGB.(img, img, img) # 32×32 Array{RGB}
    save("test" * string(i) * ".png", img)
end
