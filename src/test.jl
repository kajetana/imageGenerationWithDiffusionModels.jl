include("imageGenerationWithDiffusionModels.jl")

using .imageGenerationWithDiffusionModels
using ImageView

#filepath = "/Users/maria/Desktop/image_generation.jl/src/SyntheticImages500.mat"
const FILE_PATH = joinpath(@__DIR__, "dataset", "SyntheticImages500.mat")

data = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)  # Explicitly reference the module -

println("Top-level keys in the .mat file:")
println(keys(data))

if haskey(data, "syntheticImages")
    images = data["syntheticImages"]
    println("Images loaded successfully. Size: ", size(images))

    for i in 1:10
        img = images[:, :, 1, i]
        println("Displaying image #$i ...")
        w = imshow(img)
        sleep(1.0)              # wait 1 second so you can see it
        ImageView.close(w["gui"]["window"])    
    end
else
    println("Field 'syntheticImages' not found.")
end