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

function TResBlock(channels::Pair{<:Integer,<:Integer}, emb_dim::Int)
    TResBlock(
        Conv((3,3), channels; pad = 1),
        Conv((3,3), channels; pad = 1),
        channels[1] == channels[2] ? identity : Conv((1,1), channels),
        Dense(emb_dim, channels[2])
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

# 2x up-sampling helper
function Upsampling(channels::Pair{<:Integer, <:Integer})
    Chain(
        Upsample((2,2)),
        Conv((3,3), channels, stride = (1,1), pad = 1)
    )
    
end

# conditional skip connection block https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html
# skip connection for architectures with more than one Input
abstract type AbstractParallel end
struct ConditionalSkipConnection{T,F} <: AbstractParallel
    layers::T           #skipped layers
    connection::F       #operation which rejoins output of the skipped layers with input feature maps 
end

Flux.@functor ConditionalSkipConnection

function (skip::ConditionalSkipConnection)(x, ys...)
    skip.connection(skip.layers(x, ys...), x)
end

#ConditionalChain is basically Flux.Chain but it accepts conditional arguments(like time embeddings)



#dispatch helper
_maybe_forward(layer::AbstractParallel, x::AbstractArray, ys::AbstractArray...) = 
    layer(x, ys...)
_maybe_forward(layer::Parallel, x::AbstractArray, ys::AbstractArray...) = 
    layer(x, ys...)
_maybe_forward(layer, x::AbstractArray, ys::AbstractArray...) = 
    layer(x)

struct ConditionalChain{T<:Union{Tuple,NamedTuple}} <: AbstractParallel
    layers::T
end
Flux.@functor ConditionalChain 

ConditionalChain(xs...) = ConditionalChain(xs)          #Positional constructor
function ConditionalChain(; kw...)                      #Keyword constructor
  :layers in keys(kw) && throw(ArgumentError("a Chain cannot have a named layer called `layers`"))
  isempty(kw) && return ConditionalChain(())
  ConditionalChain(values(kw))
end

Flux.@forward ConditionalChain.layers Base.getindex, Base.length, Base.first, Base.last,
Base.iterate, Base.lastindex, Base.keys, Base.firstindex

Base.getindex(c::ConditionalChain, i::AbstractArray) = ConditionalChain(c.layers[i]...)

function (c::ConditionalChain)(x, ys...) 
    for layer in c.layers
        x = _maybe_forward(layer, x, ys...)
    end
    x
end
export TResBlock, Downsample, Upsampling, ConditionalChain, ConditionalSkipConnection
end 
