#using imageGenerationWithDiffusionModels
using Test
using Flux             
import Flux: gradient

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "blocks.jl"))                                 # module Blocks
include(joinpath(SRC, "embeddings.jl"))                             # module Embeddings
include(joinpath(SRC, "feature_encoder_network.jl"))                # make_down_path
include(joinpath(SRC, "unet.jl"))                                   # make_unet
include(joinpath(SRC, "imageGenerationWithDiffusionModels.jl"))
include(joinpath(SRC, "cosine_beta_schedule.jl"))
using .Blocks, .Embeddings, .FeatureEncoderNetwork, .UNet, .Scheduler
#TODO: Write test set for unet.jl


@testset "FeatureEncoderNetwork full path" begin
    channels = (8, 16, 32)
    emb_dim  = 24
    batch    = 2

    encoder = FeatureEncoderNetwork.make_down_path(; channels, emb_dim, in_ch = 1)

    x   = randn(Float32, 32, 32, 1, batch)
    emb = randn(Float32, emb_dim, batch)

    latent, skips = encoder.encode(x, emb)

    @test size(latent) == (4, 4, channels[end], batch)
    @test length(skips) == length(channels)
    @test all(map -> map isa Array, skips)

    # gradient through the whole encoder
    params_enc = Flux.params(encoder.encode.down_blocks) # capture blocks
    grads_enc  = gradient(() -> sum(abs2, first(encoder.encode(x, emb))),
                          params_enc)
    @test !isempty(grads_enc)
    @test all(p -> haskey(grads_enc, p), params_enc)
    @test all(p -> all(isfinite, grads_enc[p]), params_enc)
end

@testset "UNet forward / backward end-to-end" begin
    in_channels = 1
    num_levels  = 3
    model_dim   = 8
    emb_dim     = 32

    time_embed = Embeddings.LearnedTEmbedding(emb_dim)
    model = UNet.unet(in_channels, num_levels, model_dim, time_embed, emb_dim;
                      block_layer = Blocks.TResBlock,
                      num_blocks_per_level = 1)

    x = randn(Float32, 32, 32, in_channels, 2)
    t = rand(1:1000, 2)                  

    y = model(x, t)
    @test size(y) == size(x)   # UNet keeps spatial size & channel count

    gs = gradient(() -> sum(abs2, model(x, t)), Flux.params(model))
    @test !isempty(gs)
end


@testset "Blocks basic forward & backward" begin
    height = width = 16                 # spatial size of synthetic image
    batch_size   = 2                    # mini-batch
    in_channels  = 32                   # feature depth before the block
    out_channels = 64                   # feature depth after the block
    embed_dim    = 48                   # width of timestep embedding

    input_batch       = randn(Float32, height, width, in_channels, batch_size)
    timestep_embeds   = randn(Float32, embed_dim, batch_size)

    res_block = Blocks.TResBlock(in_channels => out_channels, embed_dim)

    # forward pass: the layer accepts a (height, width, in channel, batch size) batch plus a 
    # time‑embedding matrix (embedding dimension, batch size) and returns the expected shape (height, width,channel out, batch size)
    output_batch = res_block(input_batch, timestep_embeds)
    @test size(output_batch) == (height, width, out_channels, batch_size)

    # backward pass:  every parameter receives a finite gradient when back propagate loss
    grads = gradient(() -> sum(abs2, res_block(input_batch, timestep_embeds)), Flux.params(res_block))
    @test !isempty(grads)                                               # gradients exist
    @test all(p -> haskey(grads, p), Flux.params(res_block))            # every parameters hit
    @test all(p -> all(isfinite, grads[p]), Flux.params(res_block))     # all finite
end

@testset "Downsample & Upsampling" begin
    H = W = 16; ch = 8; batch = 2
    x   = randn(Float32, H, W, ch, batch)
    down = Blocks.Downsample()
    up   = Blocks.Upsampling(ch => ch)

    z = down(x)
    @test size(z) == (H ÷ 2, W ÷ 2, ch, batch)

    y = up(z)
    @test size(y) == (H, W, ch, batch)
end

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
        FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")

        data = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)
        images = data["syntheticImages"]
        img = images[:, :, 1, 1]

        # linear
        beta = LinRange(1e-4, 0.02, 500)
        alphaBar = cumprod(1 .-beta)

        @test imageGenerationWithDiffusionModels.add_noise_to_image(img, 0, alphaBar) == img

        # credits for test type: https://docs.julialang.org/en/v1/stdlib/Test/
        @test_throws ErrorException imageGenerationWithDiffusionModels.add_noise_to_image(img, 501, alphaBar)

        @test typeof(imageGenerationWithDiffusionModels.add_noise_to_image(img, 500, alphaBar)) == Tuple{Matrix{Float64}, Matrix{Float32}}
    
        @test size(imageGenerationWithDiffusionModels.add_noise_to_image(img, 500, alphaBar)[1]) == (32, 32)

        # cosine
        beta2 = imageGenerationWithDiffusionModels.cosine_beta_schedule(500)
        alphaBar2 = cumprod(1 .- beta)

        @test imageGenerationWithDiffusionModels.add_noise_to_image(img, 0, alphaBar2) == img

        @test_throws ErrorException imageGenerationWithDiffusionModels.add_noise_to_image(img, 501, alphaBar2)

        @test typeof(imageGenerationWithDiffusionModels.add_noise_to_image(img, 500, alphaBar2)) == Tuple{Matrix{Float64}, Matrix{Float32}}
    
        @test size(imageGenerationWithDiffusionModels.add_noise_to_image(img, 500, alphaBar2)[1]) == (32, 32)
    end

    @testset "add_noise_to_image_visualization" begin
        FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")

        data = imageGenerationWithDiffusionModels.load_digits_data(FILE_PATH)
        images = data["syntheticImages"]
        img = images[:, :, 1, 1]

        # linear
        beta = LinRange(1e-4, 0.02, 500)
        alphaBar = cumprod(1 .-beta)

        @test imageGenerationWithDiffusionModels.add_noise_to_image_visualization(img, 0, alphaBar) == img

        # credits for test type: https://docs.julialang.org/en/v1/stdlib/Test/
        @test_throws ErrorException imageGenerationWithDiffusionModels.add_noise_to_image_visualization(img, 501, alphaBar)

        @test typeof(imageGenerationWithDiffusionModels.add_noise_to_image_visualization(img, 500, alphaBar)) == Matrix{Float64}
    
        @test size(imageGenerationWithDiffusionModels.add_noise_to_image_visualization(img, 500, alphaBar)) == (32, 32)

        # cosine
        beta2 = imageGenerationWithDiffusionModels.cosine_beta_schedule(500)
        alphaBar2 = cumprod(1 .- beta)

        @test imageGenerationWithDiffusionModels.add_noise_to_image_visualization(img, 0, alphaBar2) == img

        @test_throws ErrorException imageGenerationWithDiffusionModels.add_noise_to_image_visualization(img, 501, alphaBar2)

        @test typeof(imageGenerationWithDiffusionModels.add_noise_to_image_visualization(img, 500, alphaBar2)) == Matrix{Float64}
    
        @test size(imageGenerationWithDiffusionModels.add_noise_to_image_visualization(img, 500, alphaBar2)) == (32, 32)
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

@testset "cosine_beta_schedule.jl" begin
    num_timesteps = 100

    @test typeof(cosine_beta_schedule(num_timesteps)) == Vector{Float64}

    @test length(cosine_beta_schedule(num_timesteps)) == 100
end
