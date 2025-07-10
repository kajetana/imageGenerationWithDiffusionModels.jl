module Scheduler

    # credits:
    # https://liorsinai.github.io/machine-learning/2022/12/29/denoising-diffusion-2-unet.html#load-data
    # https://arxiv.org/abs/2102.09672
    # https://zeta.apac.ai/en/latest/zeta/utils/cosine_beta_schedule/
    #
    # "authors found that this schedule more evenly distributes noise over the whole time range for images"
    """
        cosine_beta_schedule(num_timesteps::Int, s=0.008)

    Generates a noise schedule based on a cosine beta function.

    # Arguments
    - `num_timesteps::Int` : Number of timesteps.
    - `s` : A variable, which varies the shape of the schedule. Defaults to: `0.008`.

    # Returns
    A beta schedule.
    """
    function cosine_beta_schedule(num_timesteps::Int, s=0.008)
        t = range(0, num_timesteps; length=num_timesteps + 1)

        α_cumprods = (cos.((t / num_timesteps .+ s) / (1 + s) * π / 2)) .^ 2
        α_cumprods = α_cumprods / α_cumprods[1]

        βs = 1 .- α_cumprods[2:end] ./ α_cumprods[1:(end-1)]
        
        clamp!(βs, 0, 0.999)

        return βs
    end

    export cosine_beta_schedule
end