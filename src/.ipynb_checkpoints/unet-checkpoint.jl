module UNet
using Flux
using ..Embeddings   # for embedding dim consts
using ..FeatureEncoderNetwork: make_down_path

export make_unet

"Return a dummy callable until real model is ready."
function make_unet(; chans=(64,128,256), emb_dim=128, in_ch=1)
    down = make_down_path(; chans, emb_dim, in_ch)

    function noop_unet(x, e)   # no-op forward for now
        return similar(x)      # same shape, zeros
    end

    return noop_unet
end
end # module
