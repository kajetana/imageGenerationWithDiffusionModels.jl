module ReverseSampling
using Flux
using Random
using ImageView

function reverse_sample(model, shape::NTuple{2,Int};T::Int,
    alpha_hats::Vector{Float64}, beta_schedule::Vector{Float64}, alphas::Vector{Float64},
    noisy_image::Matrix{Float32} = nothing, noises::Vector{Matrix{Float32}} = nothing,
    istest::Bool = false
)

    # Start from Gaussian noise
    if noisy_image == nothing
        x_t = randn(Float32, shape)
    else
        x_t = noisy_image
    end

    for t in T-1:-1:1


        if istest
            eps_pred = noises[t]
        else
            eps_pred = model(x_t, t_vec)
        end

        β = beta_schedule[t]
        α = alphas[t]
        α_hat = alpha_hats[t]

        coef1 = 1 / sqrt(α)
        coef2 = β / sqrt(1 - α_hat)
        #mean = coef1 * (x_t - coef2 * eps_pred)

        mean = (x_t - sqrt(1 - α) * eps_pred) / sqrt(α)

        if t > 1
            σ = sqrt(β)
            noise = randn(Float32, shape)
            x_t = mean .+ σ .* noise
        else
            x_t = mean
        end
        x_t = mean

        #ImageView.imshow(x_t) 

    end

    return x_t
end



export reverse_sample
end
