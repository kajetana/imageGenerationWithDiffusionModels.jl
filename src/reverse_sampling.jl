module ReverseSampling
using Flux
using Random

function reverse_sample(model, shape::NTuple{4,Int};
    T::Int,
    alpha_hats::AbstractVector,)

    # Derive alphas and betas from alpha_hats
    alphas = similar(alpha_hats)
    alphas[1] = alpha_hats[1]
    for t in 2:T
        alphas[t] = alpha_hats[t] / alpha_hats[t-1]
    end
    betas = 1 .- alphas

    # Start from Gaussian noise
    x_t = randn(Float32, shape)

    for t in T:-1:1
        batch = shape[end]
        t_vec = fill(Float32(t), batch)

        eps_pred = model(x_t, t_vec)

        β = betas[t]
        α = alphas[t]
        α_hat = alpha_hats[t]

        coef1 = 1 / sqrt(α)
        coef2 = β / sqrt(1 - α_hat)
        mean = coef1 .* (x_t .- coef2 .* eps_pred)

        if t > 1
            σ = sqrt(β)
            noise = randn(Float32, shape)
            x_t = mean .+ σ .* noise
        else
            x_t = mean
        end
    end

    return x_t
end



export reverse_sample
end
