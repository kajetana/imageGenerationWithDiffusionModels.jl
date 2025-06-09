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

# TODO docs @ebektur
function add_noise_to_image(img, t, β; rng = Random.GLOBAL_RNG)
    if t==0
        return img
    end
    
    α = 1 - β[t]                                     # retain-signal factor matlab: alpha = 1 - beta(t)
    ε = randn(rng, eltype(img), size(img))           # Gaussian noise matlab: epsilon = randn(rng, size(img), 'like', img)
    return sqrt(α) .* img .+ sqrt(β[t]) .* ε         # noise the image
end

export load_digits_data, add_noise_to_image

end
