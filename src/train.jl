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

data = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH) 

data = reshape(data, 32, 32, 1, :)

###############################################################################################################
# creating the model
#
# credits: 
# https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#forward-diffusion
###############################################################################################################

model = unet(
    1,
    4,
    16,
    LearnedTEmbedding(128),
    128;
    num_blocks_per_level=1)

###############################################################################################################
# loss
#
# credits: 
# https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#forward-diffusion
###############################################################################################################


###############################################################################################################
# training the model
#
# credits: 
# https://github.com/LiorSinai/DenoisingDiffusion.jl/blob/main/examples/train_images.jl
###############################################################################################################

# no classification
training_data = Flux.DataLoader((data, ), batchsize=32, shuffle=true)

