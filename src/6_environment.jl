include("3_spot_dynamics.jl")

function step_environment!(
    S::Float64,
    vs::VolState,
    config::SimConfig,
    env:: EnvironmentState,
    rng::AbstractRNG
)
    ag_s = env.agent_state
    τ = ag_s.τ
    S_new, log_return = step_spot(S, vs, config, rng)

    if τ == 0
        env.options_completed += 1
        τ = config.T_option
    else
        τ -= 1
    end

    # update greeks

    # set spread and choose hedge

    # market interacts with quotes and a fill maybe happens

    # reward function

end
