module Embeddings
using Flux
import Flux: gelu


#Either sinusoidal or MLP will be used to generate the noise. 

"""
sinusoidal_embedding(t, dim)

Classic transformer-style Fourier features for a vector of
time-steps `t` (1-based Int or Float).  
Returns an array of size `(dim, length(t))`.

!! `dim` should be even; if odd we pad one zero row.
"""

function sinusoidal_embedding(t, dim::Integer)
    t     = Float32.(t)                    # (B,)
    half  = div(dim, 2)
    inv_f = 1f0 ./ (10000 .^ ((0:half-1) ./ half))  # (half,)
    # broadcast to (half,B)
    angles = t' .* inv_f
    emb    = vcat(sin.(angles), cos.(angles))       # (dim,B)
    if isodd(dim)
        emb = vcat(emb, zeros(Float32, 1, size(emb,2)))
    end
    return emb
end

"""
LearnedTEmbedding(dim; inner = 4dim)

Two-layer MLP that learns its own basis for the scalar time-step.

emb_layer = LearnedTEmbedding(128)
e = emb_layer(t)        # t is Vector{Int} or Vector{Float32}

Returns (dim, batch) like the sinusoidal version.
"""

# Learned MLP time-step embedding 
struct LearnedTEmbedding
    proj1::Dense
    proj2::Dense
end

"""
    LearnedTEmbedding(dim; inner = 4*dim)

Two-layer MLP that maps a scalar time-step to a length-`dim`
embedding vector.
"""
function LearnedTEmbedding(dim::Integer; inner::Integer = 4*dim)
    return LearnedTEmbedding(
        Dense(1,  inner, gelu),   # proj1
        Dense(inner, dim)         # proj2
    )
end

function (m::LearnedTEmbedding)(t::AbstractVector)
    # (1, batch)
    x = reshape(Float32.(t), 1, :)
    return m.proj2(m.proj1(x))        #  (emb_dim, batch)
end

export LearnedTEmbedding, sinusoidal_embedding
end # module