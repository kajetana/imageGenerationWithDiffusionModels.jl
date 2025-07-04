module FeatureEncoderNetwork
using Flux
import Flux: gelu

# ────────────── Conditioned Residual Block ──────────────
struct TResBlock
    conv1::Conv
    conv2::Conv
    skip::Any           # identity or 1×1 Conv
    emb_proj::Dense     # embeds time vector to out-channels
end

function TResBlock(cin::Int, cout::Int, emb_dim::Int)
    TResBlock(
        Conv((3,3), cin => cout, pad = 1),
        Conv((3,3), cout => cout, pad = 1),
        cin == cout ? identity : Conv((1,1), cin => cout),
        Dense(emb_dim, cout)
    )
end

function (m::TResBlock)(x, e)
    h = gelu.(m.conv1(x))
    emb = reshape(m.emb_proj(e), 1,1,size(h,3),size(h,4))  # (1,1,C,B)
    h .+= emb                                             # additive conditioning
    h = gelu.(m.conv2(h))
    return h .+ (m.skip === identity ? x : m.skip(x))
end

# Simple 2× down-sampling
Downsample() = MaxPool((2,2))

# ────────────── Down-path factory ──────────────
"""
make_down_path(; chans = (64,128,256), emb_dim = 128, in_ch = 1)

Builds the encoder half of a UNet.  
Returns a NamedTuple with:

* `encode(x,e)` – function running the down path and returning `(x_encoded, skips)`
* `out_ch`      – channel count after final block (needed by the bottleneck/up path)
"""
function make_down_path(; chans = (64,128,256), emb_dim = 128, in_ch = 1)
    blocks     = [TResBlock(cin, cout, emb_dim)
                  for (cin, cout) in zip((in_ch, chans[1:end-1]...), chans)]
    pool_layers = [Downsample() for _ in blocks]

    function encode(x, e)
        skips = Vector{Any}()
        for (blk, pool) in zip(blocks, pool_layers)
            x = blk(x, e); push!(skips, x)
            x = pool(x)
        end
        return x, skips
    end

    return (encode = encode,
            out_ch = chans[end])
end

export TResBlock, make_down_path
end # module