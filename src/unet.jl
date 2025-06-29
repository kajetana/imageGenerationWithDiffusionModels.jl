module UNet
using Flux
import Flux: gelu
using ..Blocks: TResBlock, ConditionalChain, ConditionalSkipConnection, Upsampling, Downsample
using ..FeatureEncoderNetwork # make_down_path
using ..Embeddings: LearnedTEmbedding, sinusoidal_embedding
"""
Dummy Unet model

Down-path  : built by `make_down_path`
Bottleneck : #TODO
Up-path    : #TODO
"""
function make_unet(; channels=(64,128,256), emb_dim=128, in_ch=1)

    # encoder
    down = make_down_path(; channels, emb_dim, in_ch)   # return tuple (encode, out_channels)
    last_channel_count = down.out_channels              

    # simple head: 1×1 Conv (latent to noise) 
    head = Conv((1,1), last_channel_count => in_ch)

    function model(img, t_emb)
        latent, skip_maps = down.encode(img, t_emb)   # forward encoder
        # TODO: place for bottleneck and decoder later 
        return head(latent)                           # noise prediction
    end

    return model
end

struct unet{E,C<:ConditionalChain}
    time_embedding::E
    chain::C
    num_levels::Int
end

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
    emb_dim = model_dim*4
    in_ch, out_ch = in_out[1]

    chain = ConditionalChain(
        init=Conv((3, 3), in_channels => in_ch, stride=(1, 1), pad=(1, 1)),
        down_1 = block_layer(in_ch => out_ch, emb_dim),
        skip_1 = ConditionalSkipConnection(
            _add_unet_level(in_out, emb_dim, 2; block_layer=block_layer, num_blocks_per_level = num_blocks_per_level),
            cat_on_channel_dim
        ),
        up_1 = block_layer((out_ch+out_ch)=>in_ch, emb_dim),
        final = Conv((3,3), in_ch => in_channels)
    )
    unet(time_embed, chain, num_levels)
end
function (u::unet)(x::AbstractArray, timesteps::AbstractVector{Int})
    emb = u.time_embedding(timesteps)
    h = u.chain(x, emb)
    h
end
Flux.@functor unet (time_embedding, chain,)
export unet
end 
