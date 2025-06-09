# Image Generation With Diffusion Models

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kajetana.github.io/imageGenerationWithDiffusionModels.jl/dev/)
[![Build Status](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

> [!WARNING]
> This project is currently under development

This Julia package implements a diffusion model to generate images of digits

![](/Screenshot%202025-06-07%20at%2011.48.08.png)

## Getting Started

Download `src/test.jl` and `src/dataset/SyntheticImages500.mat` and place them in 1 folder:

```
folder/
├─ SyntheticImages500.mat
├─ test.jl
```

Inside your `folder` install the package using the Julia REPL:

```
julia> using Pkg
julia> Pkg.add(url="https://github.com/kajetana/imageGenerationWithDiffusionModels.jl", rev="ka/FirstMileStone")
```

Run `test.jl` for a quick demo:

```
julia> include("test.jl")
```
