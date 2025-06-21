module FeatureEncoderNetwork
using Flux
import Flux: gelu
using ..Blocks: TResBlock, Downsample  

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

export make_down_path
end 

