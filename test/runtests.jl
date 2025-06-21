using imageGenerationWithDiffusionModels
using Test
using Flux             
import Flux: gradient

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "blocks.jl"))                   # module Blocks
include(joinpath(SRC, "embeddings.jl"))               # module Embeddings
include(joinpath(SRC, "feature_encoder_network.jl"))  # make_down_path
include(joinpath(SRC, "unet.jl"))                      # make_unet

using .Blocks, .Embeddings, .FeatureEncoderNetwork, .UNet
#TODO: Write test set for unet.jl

#sample timesteps --> embed them --> feed embedding plus image into the encoder/UNet
@testset "Encoder and sinusoidal embedding integration" begin
    emb_dim   = 128
    channels  = (64, 128, 256)
    batch     = 4

    # build the feature encoder (down path)
    encoder   = make_down_path(; channels, emb_dim, in_ch = 1)

    # sample timesteps and embed them
    t_steps   = [1, 500, 999, 123]                    # one batch for each timestep 
    t_emb     = sinusoidal_embedding(t_steps, emb_dim)

    # dummy image batch 32×32×1×batch_size
    x0 = randn(Float32, 32, 32, 1, batch)

    latent, skips = encoder.encode(x0, t_emb)

    # shape checks 
    @test size(latent) == (4, 4, channels[end], batch)
    @test length(skips) == length(channels)           # three stored maps as channels = (64, 128, 256)
    @test all(map -> map isa Array, skips)

    # embedding should influence the output
    t_emb_shifted = sinusoidal_embedding(t_steps .+ 1, emb_dim)
    latent_shift, _ = encoder.encode(x0, t_emb_shifted)

    @test latent != latent_shift                     # outputs differ
end

@testset "embeddings.jl" begin                    
    layer = Embeddings.LearnedTEmbedding(128)
    sinusoidal_embedding = Embeddings.sinusoidal_embedding

    @testset "sinusoidal_embedding" begin
        # shape: even dimension 
        t  = [1, 100, 500]
        d  = 128
        e  = sinusoidal_embedding(t, d)
        @test size(e) == (d, length(t))

        # shape: odd dimension (should pad one zero row) 
        d_odd = 65
        e_odd = sinusoidal_embedding(t, d_odd)
        @test size(e_odd) == (d_odd, length(t))
        @test all(e_odd[end, :] .== 0.0)          # last row is the padded zeros

        # different t give different encodings
        @test e[:, 1] != e[:, 2]               

        # deterministic: same call twice should give identical output
        @test e == sinusoidal_embedding(t, d)
    end

    @testset "LearnedTEmbedding" begin
        tb  = [1, 50, 123, 500]
        t32 = Float32[0, 250.5]

        # shape checks --------------------------------------------------
        @test size(layer(tb))  == (128, length(tb))
        @test size(layer(t32)) == (128, length(t32))

        # different timesteps → different vectors ----------------------
        e = layer(tb)
        @test e[:, 1] != e[:, 2]

        # gathers the layer’s weights
        # defines a simple scalar loss based on the layer’s output
        # asks Flux to differentiate that loss
        # important for the training and updates on the time embeddings
        θ   = Flux.params(layer)
        loss() = sum(abs2, layer(tb))
        gs  = gradient(loss, θ)
        @test all(p -> haskey(gs, p), θ)
        @test all(p -> isfinite.(gs[p]) |> all, θ)
    end

end

@testset "imageGenerationWithDiffusionModels.jl" begin

    FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")

    @testset "load_digits_data" begin
        FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")

        @test typeof(imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)) == Dict{String, Any}

        @test typeof(imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)["syntheticImages"][:, :, 1, 1]) == Matrix{Float32}
    end

    @testset "add_noise_to_image" begin
        beta = LinRange(1e-4, 0.02, 500)  # posterior variance
        alphaBar = cumprod(1 .-beta)

        FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")

        data = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)
        images = data["syntheticImages"]
        img = images[:, :, 1, 1]

        @test imageGenerationWithDiffusionModels.add_noise_to_image(img, 0, alphaBar) == img

        # credits for test type: https://docs.julialang.org/en/v1/stdlib/Test/
        @test_throws ErrorException imageGenerationWithDiffusionModels.add_noise_to_image(img, 501, alphaBar)

        @test typeof(imageGenerationWithDiffusionModels.add_noise_to_image(img, 500, alphaBar)) == Matrix{Float64}
    
        @test size(imageGenerationWithDiffusionModels.add_noise_to_image(img, 500, alphaBar)) == (32, 32)
    end

    @testset "visualize_noising_of_image" begin
        beta =  LinRange(1e-4, 0.02, 500)  # posterior variance
        alphaBar = cumprod(1 .-beta)
        ts = 500:-50:0 # noising steps

        data = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)
        images = data["syntheticImages"]
        img = images[:, :, 1, 1]

        @test typeof(imageGenerationWithDiffusionModels.visualize_noising_of_image(img, ts, alphaBar)) == Matrix{Float64}

        @test size(imageGenerationWithDiffusionModels.visualize_noising_of_image(img, ts, alphaBar)) == (32, 352)
    end

end

