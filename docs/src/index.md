```@meta
CurrentModule = imageGenerationWithDiffusionModels
```

# imageGenerationWithDiffusionModels

Documentation for [imageGenerationWithDiffusionModels](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl).

This Julia package implements a diffusion model to generate images of digits. It learns how noise alters the data and then predicts how to reverse this process to retrieve digits out of pure noise. This project can serve as a basis for more complex diffusion models.

```@index
```

```@autodocs
Modules = [
    imageGenerationWithDiffusionModels,
    imageGenerationWithDiffusionModels.FeatureEncoderNetwork,
    imageGenerationWithDiffusionModels.Scheduler,
    imageGenerationWithDiffusionModels.Embeddings,
    imageGenerationWithDiffusionModels.UNet
]
```
