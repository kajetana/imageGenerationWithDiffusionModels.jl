# Image Generation With Diffusion Models

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kajetana.github.io/imageGenerationWithDiffusionModels.jl/dev/)
[![Build Status](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kajetana/imageGenerationWithDiffusionModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kajetana/imageGenerationWithDiffusionModels.jl)

> [!WARNING]
> This project is currently under development

This Julia package implements a diffusion model to generate images of digits

![](/docs/Screenshot%202025-06-10%20at%2012.45.06.png)

The source code is structured as follows.

TODO

```
src/
├── imageGenerationWithDiffusionModels.jl       package
...
```

TODO

```
...
├── unet.jl                                     the top-level model (currently encoder-only)
├── blocks.jl                                   reusable bricks (no task-specific code)
├── feature_encoder_network.jl                  down-sampling “encoder” built from the bricks
├── embeddings.jl
├── cosine_beta_schedule.jl                     generates a noise schedule based on a cosine beta function.
├── reverse_sampling.jl                         predicts digits out with the trained model out of pure noise
...
```

TODO

```
...
├── nosing_example.jl
├── train_example.jl
...
```

TODO

```
...
├── SyntheticImages500.mat                      digit dataset
└── model.bson                                  pre-trained model
```

## Getting Started

Get acknowledged with our [Getting Started Guide](docs/GETTINGSTARTED.md) to see some use cases including visualization of the noising process as well as training of the diffusion model.

## References

"Image generation with MNIST" Article by Lior Sinai (https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#load-data)

DDPM Git Repository by ytdeepia (https://github.com/ytdeepia/DDPM/blob/main/src/training.py)

TODO
