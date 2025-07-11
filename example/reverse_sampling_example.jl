using .imageGenerationWithDiffusionModels
using ImageView
include(joinpath(@__DIR__, "../src/reverse_sampling.jl"))
using .ReverseSampling

# Load the .mat and extract one image
const FILE_PATH = joinpath(@__DIR__, "SyntheticImages500.mat")
mat    = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)
images = mat["syntheticImages"] 
img0 = images[:, :, 1, 1]          
gui = ImageView.imshow(img0)

# Noise schedule
timesteps = 1:10
num_timesteps = length(timesteps)
beta_schedule  = collect(LinRange(1e-4, 0.02, num_timesteps))
alphas = 1 .- beta_schedule
alpha_hats  = cumprod(1 .- beta_schedule)
noisy_images = Vector{Matrix{Float32}}()
noises       = Vector{Matrix{Float32}}() 

# Apply noise to the image
for t in timesteps
    if t == 1
        # Clean image: shape (32, 32)
        push!(noisy_images, Float32.(img0))  # Convert img0 to Float32
        push!(noises, zeros(Float32, 32, 32))  # Noise is all zeros
    else
        # Forward diffusion: returns (noised_img, noise)
        noised2d, noise2d = imageGenerationWithDiffusionModels.add_noise_to_image(
            Float32.(noisy_images[t-1]),  # Ensure img0 is Float32
            t,
            alpha_hats
        )
        push!(noisy_images, noised2d)
        push!(noises, noise2d)
    end
end

most_noisy_image = noisy_images[end]
gui_noisy = ImageView.imshow(most_noisy_image) 

noisy_images
noises

# Define the required arguments
model = identity  
shape = (32, 32)  
T = timesteps[end]  # Use the last timestep
noisy_image = noisy_images[end]   
istest = true  

# Call the reverse_sample function
denoised_image = ReverseSampling.reverse_sample(
    model,
    shape;
    T = T,
    alpha_hats = alpha_hats,
    beta_schedule = beta_schedule,
    alphas = alphas,
    noisy_image = noisy_image,
    noises = noises,
    istest = istest
)
# Reshape and display recovery
recovered = reshape(denoised_image, (32,32))
gui_rec = ImageView.imshow(recovered)
