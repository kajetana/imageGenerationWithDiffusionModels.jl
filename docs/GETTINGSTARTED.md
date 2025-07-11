# Getting Started Guide

Clone the repository with its contents 

_Terminal:_
```
> git clone https://github.com/kajetana/imageGenerationWithDiffusionModels.jl
> cd imageGenerationWithDiffusionModels.jl
> julia
```

and make sure to install all dependencies listed in [`Project.toml`](https://github.com/kajetana/imageGenerationWithDiffusionModels.jl/blob/main/Project.toml):

_Julia REPL:_
```
julia>]
(@v1.11) pkg> activate .
(imageGenerationWithDiffusio...) pkg> instantiate
```

Inside the `example` folder you will find 2 executable use case scenarios, which are explained more in depth in the later parts of this guide: 

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

Due to size restrictions on github, the pretrained model is accessible here: https://drive.google.com/drive/folders/1cL-ZlGzCGJ8lYINLyfVwRAx8p6VA8Ygy

Place the 'model.bson' under the 'example' folder if it does not exist. 

### Visualizing the Noising Process

![](/docs/noising.png)

Run `noising_example.jl` for a quick demo of how the noising is applied to images across different timesteps:

_Julia REPL:_
```
julia> include("example/noising_example.jl")
```

Toggle between cosine and linear schedule by changing the `cosine` variable to see the difference in the noising application:

_noising_example.jl_
```
cosine = true

if cosine
    beta = cosine_beta_schedule(num_timesteps) # cosine schedule
else
    beta = LinRange(1e-4, 0.02, num_timesteps) # linear schedule
end
```

### Training and Reverse Sampling

Run `train_example.jl` to execute the training script and see how the model predicts a digit from randomly generated noise:

_Julia REPL:_
```
julia> include("example/train_example.jl")
```

You can train the model by yourself by adjusting the training parameters to your preferences:

_training.jl:_
```
train(;FILE_PATH::String = "./example/SyntheticImages500.mat",
    num_timesteps::Int = 100,
    cosine::Bool = true,
    learning_rate::Real = 0.0001,
    epochs::Int = 15,
    batch_size::Int = 32,
    shuffle::Bool = true,
    model::unet = unet(
        1,
        5,
        16,
        LearnedTEmbedding(128),
        128;
        num_blocks_per_level=1
    ))
```

Alternatively you can use our pre-trained model `model.bson` by setting the `training` variable to `false`:

_train_example.jl:_
```
training = false

if training
    model = train(num_timesteps=num_timesteps)
else
    @load "./example/model.bson" model
end
```
