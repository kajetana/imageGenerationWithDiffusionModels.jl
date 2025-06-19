using imageGenerationWithDiffusionModels
using Test

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
