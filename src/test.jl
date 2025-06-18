#using imageGenerationWithDiffusionModels
include("imageGenerationWithDiffusionModels.jl")
using ImageView

# TODO better comment dividers

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
#
# credits for the technique of displaying different images in 1 window with ImageView: 
# https://discourse.julialang.org/t/update-existing-imshow-with-new-image-data/8296/6
###############################################################################################################

images = data["syntheticImages"]

# "dummy" image to generate the window
img = rand(32,32*11)
gui = ImageView.imshow(img)
canvas = gui["gui"]["canvas"]

# display the noising process for the first few images of the dataset
for i in 1:1
    img = images[:, :, 1, i]
    
    # TODO implement labels to the new displaying technique
    # w = imshow(hcat(frames...); name = "digit $i  (t = 500 to 0)")

    img = imageGenerationWithDiffusionModels.visualize_noising_of_image(img, ts, alphaBar)
    ImageView.imshow(canvas, img)
    sleep(4.0) 
end

ImageView.close(gui["gui"]["window"])
