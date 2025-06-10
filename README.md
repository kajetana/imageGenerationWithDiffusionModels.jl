# Image Generation With Diffusion Models

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kajetana.github.io/imageGenerationWithDiffusionModels.jl/dev/)
[![Build Status](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

> [!WARNING]
> This project is currently under development

This Julia package implements a diffusion model to generate images of digits

![](/Screenshot%202025-06-10%20at%2012.45.06.png)

## Getting Started

Download [test.jl](src/test.jl) and [SyntheticImages500.mat](src/SyntheticImages500.mat) and place them next to each other inside a `folder` of your liking:

```
folder/
├─ SyntheticImages500.mat
├─ test.jl
```

Inside your `folder` install this package using the Julia REPL and its package manager:

```
(@v1.11) pkg> activate --temp
(jl_dghlh5) pkg> add https://github.com/kajetana/imageGenerationWithDiffusionModels.jl
```

Run `test.jl` for a quick demo:

```
julia> include("test.jl")
```
