module Blocks
using Flux
import Flux: gelu

# Resblock: https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html
# 3×3 → 3×3 Residual block with time conditioning
struct TResBlock
    conv1::Conv         # 3x3
    conv2::Conv         # 3×3
    skip::Any           # identity or 1×1 conv
    emb_proj::Dense     # emb_dim → out_channel
end

function TResBlock(in_ch::Int, out_ch::Int, emb_dim::Int)
    TResBlock(
        Conv((3,3), in_ch => out_ch; pad = 1),
        Conv((3,3), out_ch => out_ch; pad = 1),
        in_ch == out_ch ? identity : Conv((1,1), in_ch => out_ch),
        Dense(emb_dim, out_ch)
    )
end

function (m::TResBlock)(x, t_emb)
    h = gelu.(m.conv1(x))
    # broadcast time embedding to (1,1,C,B) and add
    h .+= reshape(m.emb_proj(t_emb), 1,1,size(h,3),size(h,4))
    h  = gelu.(m.conv2(h))
    return h .+ (m.skip === identity ? x : m.skip(x))
end

# 2× down-sampling (MaxPool) helper
Downsample() = MaxPool((2,2))

export TResBlock, Downsample
end 
