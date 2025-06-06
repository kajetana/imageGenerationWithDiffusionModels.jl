module imageGenerationWithDiffusionModels

using MAT
using Images
using Random

function load_digits_data(filepath::String)
    matfile = matread(filepath) 
    return matfile  
end

function add_noise_to_image(img, t, β; rng = Random.GLOBAL_RNG)
    if t==0
        return img
    end
    
    α = 1 - β[t]                                     # retain-signal factor matlab: alpha = 1 - beta(t)
    ε = randn(rng, eltype(img), size(img))           # Gaussian noise matlab: epsilon = randn(rng, size(img), 'like', img)
    return sqrt(α) .* img .+ sqrt(β[t]) .* ε         # noise the image
end

end

