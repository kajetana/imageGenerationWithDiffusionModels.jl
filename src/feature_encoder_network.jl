module FeatureEncoderNetwork
using Flux
import Flux: gelu
using ..Blocks: TResBlock, Downsample, ConditionalSkipConnection, Upsample

"""
make_down_path(; channels=(64,128,256), emb_dim=128, in_ch=1)

Returns a tuple (encode, out_channels).
* encode(img, t_emb) : (latent, intermediates)
* out_channels :final channel count (needed by bottleneck)

TODO: Visualize the intermediates of MNIST
"""
function make_down_path(; channels=(64,128,256), emb_dim=128, in_ch=1)
    # build one residual block and pool per resolution level
    down_blocks = [TResBlock(prev, cur, emb_dim)
                         for (prev, cur) in zip((in_ch, channels[1:end-1]...), channels)]
    downsample_layers = [Downsample() for _ in down_blocks]

    function encode(img, t_emb)
        intermediates = Vector{Any}()
        for (block, pool) in zip(down_blocks, downsample_layers)
            img = block(img, t_emb)
            push!(intermediates, img)       # save feature map
            img = pool(img)                 # 2x down-sample
        end
        return img, intermediates           # latent and stored maps
    end

    return (encode = encode,
            out_channels = channels[end])
end

cat_on_channel_dim(h::AbstractArray, x::AbstractArray) = cat(x, h, dims=3)

"""
    _add_unet_level(
        in_out::Vector{Tuple{Int,Int}}, emb_dim::Int, level::Int;
        block_layer, num_blocks_per_level::Int, block_groups::Int, num_attention_heads::Int
    )

This function adds a level to the UNet through the use of the ConditionalSkipConnection.
The skipped layers are defined recursively until the intended level-count of the UNet has been reached.
When the break condition is reached, the middle of the Unet is created
"""
function _add_unet_level(in_out::Vector{Tuple{Int,Int}}, emb_dim::Int, level::Int;
    block_layer, num_blocks_per_level::Int, block_groups::Int, num_attention_heads::Int
)
    if level > length(in_out) # stop recursion and make the middle
        in_ch, out_ch = in_out[end]
        keys_ = (Symbol("down_$level"), :middle_1, :middle_attention, :middle_2)
        layers = (
            Conv((3, 3), in_ch => out_ch, stride=(1, 1), pad=(1, 1)),
            block_layer(out_ch => out_ch, emb_dim; groups=block_groups),
            SkipConnection(MultiheadAttention(out_ch, nhead=num_attention_heads), +),
            block_layer(out_ch => out_ch, emb_dim; groups=block_groups),
        )
    else # recurse down a layer
        in_ch_prev, out_ch_prev = in_out[level-1]
        in_ch, out_ch = in_out[level]
        down_keys = num_blocks_per_level == 1 ? [Symbol("down_$(level)")] : [Symbol("down_$(level)_$(i)") for i in 1:num_blocks_per_level]
        up_keys = num_blocks_per_level == 1 ? [Symbol("up_$(level)")] : [Symbol("up_$(level)_$(i)") for i in 1:num_blocks_per_level]
        keys_ = (
            Symbol("downsample_$(level-1)"),
            down_keys...,
            Symbol("skip_$level"),
            up_keys...,
            Symbol("upsample_$level")
        )
        down_blocks = [
            block_layer(in_ch => in_ch, emb_dim) for i in 1:num_blocks_per_level
        ]
        up_blocks = [
            block_layer((in_ch + out_ch) => out_ch, emb_dim; groups=block_groups),
            [block_layer(out_ch => out_ch, emb_dim) for i in 2:num_blocks_per_level]...
        ]
        layers = (
            Downsample(),
            down_blocks...,
            ConditionalSkipConnection(
                _add_unet_level(in_out, emb_dim, level + 1;
                    block_layer=block_layer,
                    block_groups=block_groups,
                    num_attention_heads=num_attention_heads,
                    num_blocks_per_level=num_blocks_per_level
                ),
                cat_on_channel_dim
            ),
            up_blocks...,
            Upsample(),
        )
    end
    ConditionalChain((; zip(keys_, layers)...))
end


export make_down_path
end 

