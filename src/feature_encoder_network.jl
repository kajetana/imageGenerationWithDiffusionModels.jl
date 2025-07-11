module FeatureEncoderNetwork
using Flux
import Flux: gelu
using ..Blocks: TResBlock, Downsample, ConditionalSkipConnection, Upsampling, ConditionalChain

"""
    make_down_path(; channels=(64,128,256), emb_dim=128, in_ch=1)

Returns a tuple (encode, out_channels).
* encode(img, t_emb) : (latent, intermediates)
* out_channels :final channel count (needed by bottleneck)

TODO: Visualize the intermediates of MNIST
"""
function make_down_path(; channels=(64,128,256), emb_dim=128, in_ch=1)
    # build one residual block and pool per resolution level
    down_blocks = [TResBlock(prev => cur, emb_dim)
                         for (prev, cur) in zip((in_ch, channels[1:end-1]...), channels)]
    downsample_layers = [Downsample() for _ in down_blocks]

    function encode(x, t_emb)
        #   x: the final down-sampled tensor 
        #   skips: immutable tuple of feature maps to be used as skip-connections
        skips = ()                                   # start with an empty tuple
        
        for (block, pool) in zip(down_blocks, downsample_layers)
            h   = block(x, t_emb)                    # residual / attention block
            skips = (skips..., h)                    # append new skip to tuple
            x   = pool(h)                            # 2× spatial down-sample
        end
    
        return x, skips   
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
function _add_unet_level(in_out::Vector{Tuple{Int,Int}},
                         emb_dim::Int,
                         level::Int;
                         block_layer,
                         num_blocks_per_level::Int)

    if level > length(in_out) # stop recursion and make the middle (bottleneck)
        Cin, Cout = in_out[end]

        keys   = (Symbol("downsample_$level"), :middle, Symbol("upsample_$level"))
        layers = (
            Downsample(),
            block_layer(Cout => 2Cout, emb_dim),   # middle “bottleneck”
            Upsampling(2Cout => Cout),
        )
        return ConditionalChain((; zip(keys, layers)...))
    end

    # recurse down a layer
    in_ch, out_ch = in_out[level]

    down_keys = num_blocks_per_level == 1 ?
                [Symbol("down_$level")] :
                [Symbol("down_$(level)_$i") for i in 1:num_blocks_per_level]

    up_keys = num_blocks_per_level == 1 ?
              [Symbol("up_$level")] :
              [Symbol("up_$(level)_$i") for i in 1:num_blocks_per_level]

    keys = (
        Symbol("downsample_$(level-1)"),     # comes from upper level
        down_keys...,                        
        Symbol("skip_$level"),              
        up_keys...,                          # decoder blocks
        Symbol("upsample_$level"),          
    )

    down_blocks = [
        block_layer(in_ch => out_ch, emb_dim)            # first block changes depth
        [block_layer(out_ch => out_ch, emb_dim) for _ in 2:num_blocks_per_level]...
    ]

    cat_depth   = 2out_ch                      
    up_blocks = [
        block_layer(cat_depth => out_ch, emb_dim),
        [block_layer(out_ch => out_ch, emb_dim) for _ in 2:num_blocks_per_level]...
    ]

    inner = _add_unet_level(in_out, emb_dim, level + 1;
                            block_layer = block_layer,
                            num_blocks_per_level = num_blocks_per_level)


    layers = (
        Downsample(),
        down_blocks...,
        ConditionalSkipConnection(inner, cat_on_channel_dim),
        up_blocks...,
        Upsampling(out_ch => in_ch),
    )

    ConditionalChain((; zip(keys, layers)...))
end

export make_down_path, _add_unet_level, cat_on_channel_dim
end 

