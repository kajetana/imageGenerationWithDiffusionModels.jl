using Test
@testset "blocks.jl" begin
    #generate testing data
    x = rand(Float32, 8, 8, 2, 4)
    t_emb = rand(Float32, 8, 4)

    #generate testing blocks
    inner = TResBlock(2=>2, 8)
    inner2 = TResBlock(4=>2, 8)
    skip = ConditionalSkipConnection(inner, cat_on_channel_dim)
    chain = ConditionalChain(
        skip1 = skip,
        block1 = inner2
    )
    @testset "ConditionalSkipConnection" begin
        #Get ConditionalSkipConnection output
        out = skip(x,t_emb)
        #manually execute operations of ConditionalSkipConnection to get reference output
        out_ref_skip = inner(x,t_emb)
        out_ref_skip = cat_on_channel_dim(out_ref_skip, x)

        #test if ConditionalSkipConnection output has right dimensions and values
        @test size(out) == size(out_ref_skip)
        @test out == out_ref_skip
    end
    @testset "ConditionalChain" begin
        #manually execute operations of ConditionalChain to get reference output
        out_ref_chain = skip(x,t_emb)
        out_ref_chain = inner2(out_ref_chain, t_emb)

        #Get ConditionalChain output
        out = chain(x,t_emb)

        #test if ConditionalChain output has right dimensions and values
        @test size(out) == size(out_ref_chain)
        @test out == out_ref_chain
        noise = out .+1
        function loss_fn(m)
            predicted_noise = m(x,t_emb)
            return Flux.mse(predicted_noise, noise)
        end
        optimizer = Flux.setup(Adam(0.0001), chain)
        #Test if gradients can be comuputet
        loss_val, grads = Flux.withgradient(loss_fn, chain)
        @test !isnothing(grads) && !isnothing(loss_val)
        @test isfinite(loss_val)
        #Test if update! works on ConditionalChain and ConditionalSkipConnection
        @test_nowarn Flux.update!(optimizer, chain, grads[1])
    end
end
