module UNet
using Flux
import Flux: gelu
using ..Blocks: TResBlock, ConditionalChain, ConditionalSkipConnection, Upsampling, Downsample
using ..FeatureEncoderNetwork # make_down_path
using ..Embeddings: LearnedTEmbedding, sinusoidal_embedding

"""
    unet{E,C<:ConditionalChain}
Struct of the UNet Model.
"""
struct unet{E,C<:ConditionalChain}
    time_embedding::E
    chain::C
    num_levels::Int
end
"""
    unet(
        in_channels::Int,
        num_levels::Int,
        model_dim::Int,
        time_embed,
        emb_dim::Int;
        block_layer=TResBlock,
        num_blocks_per_level::Int=1,
    )
Constructs a UNet Model.

The depth of the Model is parametrized by `num_levels`. `time_embed` is a time embedding block with an output length `emb_dim`.
"""
function unet(
        in_channels::Int,
        num_levels::Int,
        model_dim::Int,
        time_embed,
        emb_dim::Int;
        block_layer=TResBlock,
        num_blocks_per_level::Int=1,
    )
    channels = [model_dim * 2^i for i in 0:num_levels-1]
    in_out = collect(zip(channels[1:end-1], channels[2:end]))
    in_ch, out_ch = in_out[1]

    chain = ConditionalChain(
        init=Conv((3, 3), in_channels => in_ch, stride=(1, 1), pad=(1, 1)),
        down_1 = block_layer(in_ch => out_ch, emb_dim),
        skip_1 = ConditionalSkipConnection(
            _add_unet_level(in_out, emb_dim, 2; block_layer=block_layer, num_blocks_per_level = num_blocks_per_level),
            cat_on_channel_dim
        ),
        up_1 = block_layer((out_ch+out_ch)=>in_ch, emb_dim),
        final = Conv((3,3), in_ch => in_channels, pad = (1,1))
    )
    unet(time_embed, chain, num_levels)
end
"""
    (u::unet)(x::AbstractArray, timesteps::AbstractVector)
Forward pass for the UNet model.

Usually takes an Image in form of a Vector and a timestep as inputs and returns an Image in form of a vector.
"""
function (u::unet)(x::AbstractArray, timesteps::AbstractVector)
    emb = u.time_embedding(timesteps)
    h = u.chain(x, emb)
    h
end
Flux.@functor unet (time_embedding, chain,)
export unet
end 
