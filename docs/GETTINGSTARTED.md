# Getting Started Guide

Download the [`example`](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/tree/main/example) folder with its contents, install this package using the Julia REPL and its package manager:

```
(@v1.11) pkg> activate --temp
(jl_dghlh5) pkg> add https://github.com/kajetana/imageGenerationWithDiffusionModels.jl
```

and install all dependencies listed in [`Project.toml`](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/blob/main/Project.toml):

```
BSON = "0.3.9"
Flux = "0.16.4"
ImageView = "0.12.6"
Images = "0.26.2"
MAT = "0.10.7"
Random = "1.11.0"
julia = "1.11"
```

Inside the downloaded `example` folder you will find 2 executable use case scenarios, which are explained more in depth in the following sections:

```
example/
├── nosing_example.jl                           visualization of the noising application process      
├── train_example.jl                            model training
...
```

as well as our base dataset [(source)](https://webhomes.maths.ed.ac.uk/~dhigham/SRpaper.zip) and a pre-trained model:

```
...
├── SyntheticImages500.mat                      digit dataset
└── model.bson                                  pre-trained model
```

### Visualizing the Noising Process

![](/docs/Screenshot%202025-06-10%20at%2012.45.06.png)

Run `noising_example.jl` for a quick demo of how the noising is applied to images across different timesteps:

```
julia> include("noising_example.jl")
```

### Training and Reverse Sampling

You can train the model by yourself by adjusting the training variables:

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
