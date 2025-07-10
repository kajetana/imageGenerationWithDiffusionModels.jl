# Image Generation With Diffusion Models

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kajetana.github.io/imageGenerationWithDiffusionModels.jl/dev/)
[![Build Status](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kajetana/imageGenerationWithDiffusionModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kajetana/imageGenerationWithDiffusionModels.jl)

> [!WARNING]
> This project is currently under development

This Julia package implements a diffusion model to generate images of digits. It learns how noise alters the data and then predicts how to reverse this process to retrieve digits out of pure noise. This project can serve as a basis for more complex diffusion models. 

![](/docs/noising.png)

## Source Code

The source code is structured as follows.

The main module, which combines all the core components into a package:

```
src/
├── imageGenerationWithDiffusionModels.jl       main module of the package
...
```

Model architecture components and utility functions:

```
...
├── unet.jl                                     the top-level model (currently encoder-only)
├── blocks.jl                                   reusable bricks (no task-specific code)
├── feature_encoder_network.jl                  down-sampling “encoder” built from the bricks
├── embeddings.jl                               embeddings
├── training.jl                                 trains the model
├── cosine_beta_schedule.jl                     generates a noise schedule based on a cosine beta function
└── reverse_sampling.jl                         reverses the diffusion process
```

## Getting Started

Get acknowledged with our 

- [Getting Started Guide](docs/GETTINGSTARTED.md) 

to see some use cases including visualization of the noising process as well as training of the diffusion model.

## References

> "Image generation with MNIST" Article by Lior Sinai (https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#load-data)

> "DenoisingDiffusion.jl" GitHub Repository by Lior Sinai (https://github.com/LiorSinai/DenoisingDiffusion.jl)

> "DDPM" GitHub Repository by ytdeepia (https://github.com/ytdeepia/DDPM)

> "Generate Images Using Diffusion" Example by MatLab (https://de.mathworks.com/help/deeplearning/ug/generate-images-using-diffusion.html)
