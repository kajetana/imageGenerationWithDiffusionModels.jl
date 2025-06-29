module imageGenerationWithDiffusionModels

using MAT
using Images
using Random
using Flux

# TODO default filepath?
"""
    load_digits_data(filepath::String)

Loads digits data from a .mat `filepath`.

# Arguments
- `filepath::String`: A filepath to .mat digits data.
...
"""
function load_digits_data(filepath::String)
    return matread(filepath)  
end

"""
    add_noise_to_image(img::Vector{Float64}, noise_step::Int64, alpha_bar::Vector{Float64}; rng = Random.GLOBAL_RNG)

Applies Gaussian noise to an image.

# Arguments
- `img::Matrix{Float32}` : Input image
- `noise_step::Int64` : A noising step
- `alpha_bar::Vector{Float64}` : Vector of noise parameters. Length must be at least "noise_step". Comupted by taking the Cumulative product of (1-"variance schedule")
- `rng::Random.TaskLocalRNG`: Random number generator. Defaults to: `Random.GLOBAL_RNG`.

# Returns
A noised version of image.
"""
function add_noise_to_image(img, noise_step, alpha_bar, rng = Random.GLOBAL_RNG)
    if noise_step == 0
        return img
    end

    if noise_step > length(alpha_bar)
        error()
    end
    
    sqrtOneMinusalphaBar = sqrt(1 - alpha_bar[noise_step])                        # TODO        
    z = randn(rng, eltype(img), size(img))                                        # TODO
    return sqrt(alpha_bar[noise_step]).* img .+ sqrtOneMinusalphaBar .* z         # noise the image
end

"""
    visualize_noising_of_image(img, noise_step, alpha_bar, rng = Random.GLOBAL_RNG)

Visualizes the Gaussian noising process of an image.

# Arguments
- `img::Matrix{Float32}` : Input image
- `noise_step::StepRange{Int64, Int64}` : Number of noising steps (in each step, noise is applied to the output of the previous step)
- `alpha_bar::Vector{Float64}` : Vector of noise parameters. Length must be at least "noise_step". Comupted by taking the Cumulative product of (1-"variance schedule")
- `rng`: Random number generator. Defaults to: `Random.GLOBAL_RNG`.

# Returns
An image visualizing the Gaussian noising process of an image horizontally.
"""
function visualize_noising_of_image(img, noise_step, alpha_bar, rng = Random.GLOBAL_RNG)
    return hcat([imageGenerationWithDiffusionModels.add_noise_to_image(img, t, alpha_bar, rng) for t in noise_step]...)
end


=======
# credits: 
# https://github.com/LiorSinai/DenoisingDiffusion.jl/blob/main/examples/train_images.jl
# https://fluxml.ai/Flux.jl/previews/PR1786/data/dataloader/\
# https://adrianhill.de/julia-ml-course/L7_Deep_Learning/
"""
    preprocess_data(data::Matrix{Float32}, batch_size::Int, shuffle::Boolean)

Preprocesses data and returns a Flux object for iteration over "mini-batches" of data for diffusion model training.

# Arguments
- `data::Matrix{Float32}` : Digit data
- `batch_size::Int` : Number of training records per batch.  Defaults to: `1`, like Flux's default behaviour
- `shuffle::Boolean` : Controls wheter to shuffle data. Defaults to: `true`, like Flux's default behaviour

# Returns
A Flux object for iteration over "mini-batches" of data
"""
function preprocess_data(data::Matrix{Float32}, batch_size::Int=1, shuffle::Boolean=true)
    data = reshape(data, 32, 32, 1, :)

    # no classification
    return Flux.DataLoader((data, ), batchsize=batch_size, shuffle=shuffle)
end

include("embeddings.jl")
using .Embeddings
include("blocks.jl")
using .Blocks
include("feature_encoder_network.jl")
using .FeatureEncoderNetwork
include("unet.jl")
using .UNet

export load_digits_data, add_noise_to_image, visualize_noising_of_image, _add_unet_level, TResBlock, unet, LearnedTEmbedding, sinusoidal_embedding


end
