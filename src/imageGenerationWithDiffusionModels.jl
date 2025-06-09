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

# TODO docs 
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
