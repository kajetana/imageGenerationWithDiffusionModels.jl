module imageGenerationWithDiffusionModels

using MAT
using Images
using Random

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

Applies Gaussian noise to an Image.

# Arguments
- `img::Vector{Float64}` : Input image
- `noise_step::Int64` : Number of noising steps (in each step, noise is applied to the output of the previous step)
- `alpha_bar::Vector{Float64}` : Vector of noise parameters. Length must be at least "noise_step". Comupted by taking the Cumulative product of (1-"variance schedule")
- `rng`: Random number generator. Defaults to: `Random.GLOBAL_RNG`.

# Returns
A noised version of `img`.
"""
function add_noise_to_image(img, noise_step, alpha_bar; rng = Random.GLOBAL_RNG)
    if noise_step==0
        return img
    end
    
    sqrtOneMinusalphaBar = sqrt(1 - alpha_bar[noise_step])                                
    z = randn(rng, eltype(img), size(img))          
    return sqrt(alpha_bar[noise_step]).* img .+ sqrtOneMinusalphaBar .* z         # noise the image
end

export load_digits_data, add_noise_to_image

end
