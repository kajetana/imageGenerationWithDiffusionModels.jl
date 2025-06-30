module Scheduler

    # credits:
    # https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#load-data
    function cosine_beta_schedule(num_timesteps::Int, s=0.008)
        t = range(0, num_timesteps; length=num_timesteps + 1)
        α_cumprods = (cos.((t / num_timesteps .+ s) / (1 + s) * π / 2)) .^ 2
        α_cumprods = α_cumprods / α_cumprods[1]
        βs = 1 .- α_cumprods[2:end] ./ α_cumprods[1:(end-1)]
        clamp!(βs, 0, 0.999)

        return βs
    end
end