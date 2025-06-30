using .imageGenerationWithDiffusionModels
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
for i in 1:4
    img = images[:, :, 1, i]
    
    # TODO implement labels to the new displaying technique
    # w = imshow(hcat(frames...); name = "digit $i  (t = 500 to 0)")

    img = imageGenerationWithDiffusionModels.visualize_noising_of_image(img, ts, alphaBar)
    ImageView.imshow(canvas, img)
    sleep(4.0) 
end

ImageView.close(gui["gui"]["window"])


T = 500
betas, alphas, alpha_bar = imageGenerationWithDiffusionModels.get_schedule(T)

###############################################################################################################
# Reverse sampling: generating new images
###############################################################################################################
# 1. Load or define your model (UNet from your module)
# model = imageGenerationWithDiffusionModels.unet(in_channels=1, out_channels=1, ch=32, emb_dim=32, ch_mult=[1,2], num_res_blocks=1)

# 2. Reverse sample
# samples = ReverseSampling.reverse_sample(model, (32, 32, 1, 8); T=T, betas=betas, embedding_fn=embedding_fn)

# 3. Postprocess samples to 2D images and visualize
# samples = reshape(samples, 32, 32, 8)  # drop channel dim
# img = hcat([samples[:, :, i] for i in 1:8]...)
# ImageView.imshow(canvas, img)
# sleep(6.0)
# ImageView.close(gui["gui"]["window"])


