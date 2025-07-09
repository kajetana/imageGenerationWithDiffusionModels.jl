module Blocks
using Flux
import Flux: gelu

# Resblock: https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html
# 3×3 → 3×3 Residual block with time conditioning
abstract type AbstractParallel end
"""
    struct TResBlock<: AbstractParallel
A Residual block with multiple inputs (conditional inputs)
"""
struct TResBlock<: AbstractParallel
    conv1::Conv         # 3x3
    conv2::Conv         # 3×3
    skip::Any           # identity or 1×1 conv
    emb_proj::Dense     # emb_dim → out_channel
end
Flux.Flux.@functor TResBlock
"""
    TResBlock(channels::Pair{<:Integer,<:Integer}, emb_dim::Int)
Constructs a `TResBlock` with in- and output `channels`.
"""
function TResBlock(channels::Pair{<:Integer,<:Integer}, emb_dim::Int)
    TResBlock(
        Conv((3,3), channels; pad = 1),
        Conv((3,3), channels[2]=>channels[2]; pad = 1),
        channels[1] == channels[2] ? identity : Conv((1,1), channels),
        Dense(emb_dim, channels[2])
    )
end
"""
    (m::TResBlock)(x, t_emb)
Forward pass for the `TResBlock`.
A `SkipConnection` skips over two convolution layers. `x` is the layer input and t_emb in added to the feature maps as a bias after the first convolution.
See also `Flux.SkipConnection`, `Flux.Conv`
"""
function (m::TResBlock)(x, t_emb)
    #@info "Feature shape before failing conv: ", size(x)
    h = gelu.(m.conv1(x))
    # broadcast time embedding to (1,1,C,B) and add
    h = h .+ reshape(m.emb_proj(t_emb), 1,1,size(h,3),size(h,4))
    h  = gelu.(m.conv2(h))
    return h .+ (m.skip === identity ? x : m.skip(x))
end

# 2× down-sampling (MaxPool) helper
"""
    Downsample()
Applies a 2x2 MaxPooling to the Input.
"""
Downsample() = MaxPool((2,2))

# 2x up-sampling helper
"""
    Upsampling(channels::Pair{<:Integer, <:Integer})
Applies a 2x2 Upsampling and a 3x3 padded Convolution parametrized by `channels`
"""
function Upsampling(channels::Pair{<:Integer, <:Integer})
    Chain(
        Upsample((2,2)),
        Conv((3,3), channels, stride = (1,1), pad = 1)
    )
    
end

# conditional skip connection block https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html
# skip connection for architectures with more than one Input
"""
    ConditionalSkipConnection{T,F} <: AbstractParallel
A skip connection, that accepts multiple inputs (conditional inputs).

The struct holds `layers` that are meant to be skipped and `connection` which rejoins the output of the skipped layers with the input.
See also `Flux.SkipConnection`
"""
struct ConditionalSkipConnection{T,F} <: AbstractParallel
    layers::T           #skipped layers
    connection::F       #operation which rejoins output of the skipped layers with input feature maps 
end

Flux.@functor ConditionalSkipConnection
"""
    (skip::ConditionalSkipConnection)(x, ys...)
Forward pass for the ConditionalSkipConnection.

`layers` is applied to `xs` and `ys`. The output is then fused with `x` by applying `connection` to them.
"""
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
"""
    ConditionalChain{T<:Union{Tuple,NamedTuple}} <: AbstractParallel
A Chain operator, that accepts multiple inputs (conditional inputs).
See also `Flux.Chain`
"""
struct ConditionalChain{T<:Union{Tuple,NamedTuple}} <: AbstractParallel
    layers::T
end
Flux.@functor ConditionalChain 
"""
    ConditionalChain(xs...)
Constructs a `ConditionalChain` object from the layers `xs...`
"""
ConditionalChain(xs...) = ConditionalChain(xs)          #Positional constructor
"""
    ConditionalChain(; kw...)
Keyword constructor for the ConditionalChain.
"""
function ConditionalChain(; kw...)                      #Keyword constructor
  :layers in keys(kw) && throw(ArgumentError("a Chain cannot have a named layer called `layers`"))
  isempty(kw) && return ConditionalChain(())
  ConditionalChain(values(kw))
end

Flux.@forward ConditionalChain.layers Base.getindex, Base.length, Base.first, Base.last,
Base.iterate, Base.lastindex, Base.keys, Base.firstindex

Base.getindex(c::ConditionalChain, i::AbstractArray) = ConditionalChain(c.layers[i]...)
"""
    (c::ConditionalChain)(x, ys...)
Forward operator for the `ConditionalChain`
The Layers of the Chain are applied to either only `x` or both `x` and `ys...` depending on the number of inputs, that the respective layer accepts.
"""
function (c::ConditionalChain)(x, ys...) 
    for layer in c.layers
        x = _maybe_forward(layer, x, ys...)
    end
    x
end
export TResBlock, Downsample, Upsampling, ConditionalChain, ConditionalSkipConnection
end 
