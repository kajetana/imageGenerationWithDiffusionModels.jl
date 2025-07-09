using Test
using Flux             
import Flux: gradient
using Random
using imageGenerationWithDiffusionModels

@testset "FeatureEncoderNetwork full path" begin
    channels = (8, 16, 32)
    emb_dim  = 24
    batch    = 2
    #encoder = FeatureEncoderNetwork.make_down_path(; channels, emb_dim, in_ch = 1)
    encoder = make_down_path(; channels, emb_dim, in_ch = 1)

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

    time_embed = LearnedTEmbedding(emb_dim)
    time_embed = LearnedTEmbedding(emb_dim)
    model = unet(in_channels, num_levels, model_dim, time_embed, emb_dim;
                      block_layer = TResBlock,
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

    #res_block = Blocks.TResBlock(in_channels => out_channels, embed_dim)
    res_block = TResBlock(in_channels => out_channels, embed_dim)

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
    layer = LearnedTEmbedding(128)

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

@testset "training.jl" begin
    FILE_PATH = joinpath(@__DIR__, "", "SyntheticImages500.mat")

    model = train(num_timesteps=1, batch_size=500, epochs=1, FILE_PATH=FILE_PATH)

    # linear
    #model2 = train(num_timesteps=1, batch_size=500, epochs=1, cosine=false)

    @test typeof(model) == unet{LearnedTEmbedding, ConditionalChain{@NamedTuple{init::Flux.Conv{2, 2, typeof(identity), Array{Float32, 4}, Vector{Float32}}, down_1::TResBlock, skip_1::ConditionalSkipConnection{ConditionalChain{@NamedTuple{downsample_1::Flux.MaxPool{2, 4}, down_2::TResBlock, skip_2::ConditionalSkipConnection{ConditionalChain{@NamedTuple{downsample_2::Flux.MaxPool{2, 4}, down_3::TResBlock, skip_3::ConditionalSkipConnection{ConditionalChain{@NamedTuple{downsample_3::Flux.MaxPool{2, 4}, down_4::TResBlock, skip_4::ConditionalSkipConnection{ConditionalChain{@NamedTuple{downsample_5::Flux.MaxPool{2, 4}, middle::TResBlock, upsample_5::Flux.Chain{Tuple{Flux.Upsample{:nearest, Tuple{Int64, Int64}, Nothing}, Flux.Conv{2, 4, typeof(identity), Array{Float32, 4}, Vector{Float32}}}}}}, typeof(cat_on_channel_dim)}, up_4::TResBlock, upsample_4::Flux.Chain{Tuple{Flux.Upsample{:nearest, Tuple{Int64, Int64}, Nothing}, Flux.Conv{2, 4, typeof(identity), Array{Float32, 4}, Vector{Float32}}}}}}, typeof(cat_on_channel_dim)}, up_3::TResBlock, upsample_3::Flux.Chain{Tuple{Flux.Upsample{:nearest, Tuple{Int64, Int64}, Nothing}, Flux.Conv{2, 4, typeof(identity), Array{Float32, 4}, Vector{Float32}}}}}}, typeof(cat_on_channel_dim)}, up_2::TResBlock, upsample_2::Flux.Chain{Tuple{Flux.Upsample{:nearest, Tuple{Int64, Int64}, Nothing}, Flux.Conv{2, 4, typeof(identity), Array{Float32, 4}, Vector{Float32}}}}}}, typeof(cat_on_channel_dim)}, up_1::TResBlock, final::Flux.Conv{2, 2, typeof(identity), Array{Float32, 4}, Vector{Float32}}}}}

    # credits: https://www.jlhub.com/julia/manual/en/function/isfile
    #@test isfile(joinpath(@__DIR__, "model.bson"))
end

# @testset "reverse_sampling.jl" begin
#     shape = (1, 28, 28, 4)  # channels, height, width, batch
#     T = 5
#     alpha_hats = Float32.([0.9^t for t in 1:T])  # geometric decay

#     @testset "Output shape and type" begin
#         x_sampled = reverse_sample(mock_model_zeros, shape; T=T, alpha_hats=alpha_hats)
#         @test size(x_sampled) == shape
#         @test eltype(x_sampled) == Float32
#     end

#     @testset "Runs without error (identity model)" begin
#         x_sampled = reverse_sample(mock_model_identity, shape; T=T, alpha_hats=alpha_hats)
#         @test !any(isnan, x_sampled)
#     end

#     @testset "Edge case: T = 1" begin
#         alpha_hats_edge = Float32.([0.95])
#         x_sampled = reverse_sample(mock_model_zeros, shape; T=1, alpha_hats=alpha_hats_edge)
#         @test size(x_sampled) == shape
#     end

#     @testset "All-zero model output -> Gaussian diffusion" begin
#         Random.seed!(42)
#         x1 = reverse_sample(mock_model_zeros, shape; T=T, alpha_hats=alpha_hats)
#         Random.seed!(42)
#         x2 = reverse_sample(mock_model_zeros, shape; T=T, alpha_hats=alpha_hats)
#         @test x1 == x2  # deterministic if model and RNG fixed
#     end
# end