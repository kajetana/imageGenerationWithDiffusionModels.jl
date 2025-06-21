module UNet
using Flux
import Flux: gelu
using ..Blocks                # TResBlock, Downsample
using ..FeatureEncoderNetwork # make_down_path

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
end 
