using imageGenerationWithDiffusionModels
using ImageView

const FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")
beta =  LinRange(1e-4, 0.02, 500)  #posterior variance
alphaBar = cumprod(1 .-beta)
ts = 500:-50:0 #noise step                      
data = load_digits_data(FILE_PATH)  # Explicitly reference the module 

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
                
        #Add noise to the image
        #for each number, plot 11 images from t=500 to t=0 next to each other         
        frames = [imageGenerationWithDiffusionModels.add_noise_to_image(img, t, alphaBar) for t in ts]
        w = imshow(hcat(frames...); name = "digit $i  (t = 500 to 0)")
        sleep(2.0)
        ImageView.close(w["gui"]["window"])

    end
else
    println("Field 'syntheticImages' not found.")
end
