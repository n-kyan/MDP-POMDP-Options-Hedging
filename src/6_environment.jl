include("3_spot_dynamics.jl")

# Initialize a fresh episode.
# Resets portfolio and environment; resets particle filter to prior.
# Returns the initial AgentState.
function initialize_episode!(
    env::EnvironmentState,
    portfolio::Portfolio,
    vm::VolModel,
    config::SimConfig,
    pf::ParticleFilter,
)
    portfolio.cash   = 0.0
    portfolio.q_spot = 0.0
    empty!(portfolio.option_quantities)

    env.vol_state         = VolState(vm)
    env.options_completed = 0

    empty!(env.current_options)
    push!(env.current_options, OptionContract(round(config.S0), true))
    push!(portfolio.option_quantities, 0)

    τ = config.T_option * config.Δt

    reset!(pf)
    σ_hat_val = get_σ_hat(pf)

    initial_state = build_agent_state(
        portfolio, env.current_options, config.S0, τ, σ_hat_val, config.r, NaN, 0
    )
    env.agent_state = initial_state

    return initial_state
end

# Single environment step.
#
# Sequence:
#   1. Quote around believed V̂ with half-spread δ
#   2. Fills simulated against true V* (fill asymmetry is the POMDP signal)
#   3. Execute hedge: adjust q_spot to target Δ_target
#   4. Spot moves (hidden regime may switch)
#   5. Option expiry check and rollover
#   6. Particle filter update on observed log-return
#   7. Reward = true wealth change − φ·Δ_target²
#
# Returns (next_state, reward, done, StepInfo)
function step_environment!(
    env::EnvironmentState,
    portfolio::Portfolio,
    pf::ParticleFilter,
    action::MarketMakingAction,
    config::SimConfig,
    rng::AbstractRNG;
    σ_hat_override::Float64 = NaN,
)
    s    = env.agent_state
    vs   = env.vol_state
    S    = s.S
    τ    = s.τ
    opt  = env.current_options[1]
    δ    = action.δ
    Δ_target = action.Δ_target

    # 1. Believed fair value and quotes
    # Oracle benchmarks pass σ_hat_override = transition-weighted σ; RL agents use particle filter.
    σ_hat_val = isnan(σ_hat_override) ? get_σ_hat(pf) : σ_hat_override
    bs_hat    = bs_all(S, opt.K, τ, σ_hat_val, config.r; call = opt.is_call)
    hat_V     = bs_hat.price
    bid_price = hat_V - δ
    ask_price = hat_V + δ

    # 2. True market price (for fill evaluation and wealth tracking)
    market_belief = perfect_regime_belief(vs)
    V_market = bs_all_belief_weighted(
        S, opt.K, τ, vs.vm.σ_levels, market_belief, config.r; call = opt.is_call
    ).price

    # 3. True wealth before action (uses market V*, not agent's hat_V)
    wealth_before = compute_true_wealth(portfolio, env.current_options, S, τ, vs, config.r)

    # 4. Simulate fills against V_market
    fill = simulate_fills(bid_price, ask_price, V_market, config, rng)

    # 5. Update inventory and cash from fills
    update_from_fills!(portfolio, fill, 1)

    # 6. Execute hedge
    # Re-compute hat_Δ_P post-fill (inventory may have changed)
    q_post = portfolio.option_quantities[1]
    hat_Δ_post_fill = q_post * bs_hat.Δ + portfolio.q_spot
    shares_traded, hedge_cost = execute_hedge!(portfolio, Δ_target, hat_Δ_post_fill, S, config)

    # 7. Spot moves; vol regime may switch
    S_new, vs_new, log_return = step_spot(S, vs, config, rng)
    τ_new = τ - config.Δt

    # 8. Option expiry and rollover
    done = false
    if τ_new < config.Δt / 2
        reset_for_new_option!(portfolio, env.current_options, S_new)
        env.options_completed += 1
        τ_new = config.T_option * config.Δt
        if env.options_completed >= config.n_options_per_episode
            done = true
        end
    end

    # 9. Update particle filter on observed return
    update!(pf, log_return, config)

    # 10. True wealth after spot move (and after any expiry settlement)
    wealth_after = compute_true_wealth(portfolio, env.current_options, S_new, τ_new, vs_new, config.r)

    # 11. Reward: true P&L minus risk penalty on chosen exposure
    reward = compute_reward(wealth_before, wealth_after, Δ_target, config)

    # 12. Build next agent state
    σ_hat_new = if isnan(σ_hat_override)
        get_σ_hat(pf)
    else
        # Oracle mode: transition-weighted σ for the new vol state
        ws = perfect_regime_belief(vs_new)
        sqrt(sum(ws .* vs_new.vm.σ_levels .^ 2))
    end
    next_state = build_agent_state(
        portfolio, env.current_options, S_new, τ_new, σ_hat_new, config.r, log_return, fill.f_t
    )

    env.agent_state = next_state
    env.vol_state   = vs_new

    step_info = StepInfo(log_return, fill, shares_traded, hedge_cost, wealth_before, wealth_after)

    return next_state, reward, done, step_info
end
