# Getting Started Guide

Inside a `folder` of your liking, install this package using the Julia REPL and its package manager:

```
(@v1.11) pkg> activate --temp
(jl_dghlh5) pkg> add https://github.com/kajetana/imageGenerationWithDiffusionModels.jl#encoder
```

and install all dependencies listed in `Project.toml`:

```
BSON = "0.3.9"
Flux = "0.16.4"
ImageView = "0.12.6"
Images = "0.26.2"
MAT = "0.10.7"
Random = "1.11.0"
julia = "1.11"
```

Download [noising_example.jl](src/test.jl), [train_example.jl](src/train.jl) and [SyntheticImages500.mat](src/SyntheticImages500.mat) and place them next to each other inside a `folder` of your liking:

```
folder/
├─ SyntheticImages500.mat
├─ noising_example.jl
├─ train_example.jl
```

### Visualizing the Noising Process

Run `noising_example.jl` for a quick demo of how the noising is applied to images across different timesteps:

```
julia> include("noising_example.jl")
```

### Training and Reverse Sampling

You can train the model by yourself, adjust the training variables:

```
# training variables
learning_rate = 0.001
epochs = 5
batch_size = 32
shuffle = true
```

as well as the noising variables to your preferences. You can also choose between applying a linear or cosine noise schedule:

```
# noising variables
num_timesteps = 100
beta = imageGenerationWithDiffusionModels.cosine_beta_schedule(num_timesteps) # cosine schedule
#beta =  LinRange(1e-4, 0.02, 100) # linear schedule
alphaBar = cumprod(1 .- beta)
```

Alternatively you can use our pre-trained model `model.bson` by setting the `training` variable to `false`:

```
training = false
```

Run `train_example.jl` to execute the training script and see how the model predicts a digit from randomly generated noise:

```
julia> include("train_example.jl")
```