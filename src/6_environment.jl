include("3_spot_dynamics.jl")

function initialize_episode!(
    env::EnvironmentState,
    portfolio::Portfolio,
    vm::VolModel,
    config::SimConfig;
    level::Int = 2  # 1 = constant vol, 2 = known regime, 3 = hidden regime
)
    # Reset portfolio to zero
    portfolio.cash   = 0.0
    portfolio.q_spot = 0.0
    empty!(portfolio.option_quantities)

    # Fresh vol state (regime sampled from stationary distribution to reflect long run time spent in each regime)
    env.vol_state = VolState(vm)
    env.options_completed = 0

    # Add ATM call at S0 to the "menu" of tradable assets
    empty!(env.current_options)
    push!(env.current_options, OptionContract(round(config.S0), true))
    # # Add ATM put at S0 to the "menu" of tradable assets
    # push!(env.current_options, OptionContract(round(config.S0), false)) 
    push!(portfolio.option_quantities, 0)

    # Initial τ in years: T_option trading days × Δt years/day
    τ = config.T_option * config.Δt

    # Initial regime belief
    n_regimes = length(vm.σ_levels)
    regime_belief = compute_initial_belief(env.vol_state, n_regimes, level)

    # Build initial AgentState
    σ_regimes = vm.σ_levels
    initial_state = build_agent_state(
        portfolio, env.current_options, config.S0, τ, regime_belief, σ_regimes, config.r
    )
    env.agent_state = initial_state

    return initial_state   
end

# Computes the agent's initial regime belief based on the level of the simulation.
function compute_initial_belief(vs::VolState, n_regimes::Int, level::Int)
    if n_regimes == 1 || level == 1
        return [1.0]
    elseif level == 2
        # Perfect observability of true starting regime
        return [vs.regime_idx == i ? 1.0 : 0.0 for i in 1:n_regimes]
    else
        # Level 3: start from uniform dist over n_regimes
        return fill(1.0/n_regimes, n_regimes)
    end
end

# belief update router based on vol visivility level
function update_belief(
    belief_old::Vector{Float64},
    vs_new::VolState,
    n_regimes::Int,
    level::Int,
    log_return::Float64,
    fill::FillOutcome,
    options::Vector{OptionContract},
    S_new::Float64,
    τ_new::Float64,
    config::SimConfig,
    belief_update_fn
)
    if n_regimes == 1 || level == 1
        return [1.0]

    elseif belief_update_fn === nothing
        # Level 2: perfect observability → one-hot on true regime
        return [vs_new.regime_idx == i ? 1.0 : 0.0 for i in 1:n_regimes]

    else
        # Level 3: Hamilton filter (Module 12)
        return belief_update_fn(belief_old, log_return, fill, options, S_new, τ_new, config)
    end
end

# Main step function
function step_environment!(
    env::EnvironmentState,
    portfolio::Portfolio,
    action::MarketMakingAction,
    config::SimConfig,
    rng::AbstractRNG;
    belief_update_fn = nothing,
    level::Int = 2
)
    # 1. Unpack Current State
    agent        = env.agent_state
    vs        = env.vol_state
    S         = agent.S
    τ         = agent.τ
    vm        = vs.vm
    σ_regimes = vm.σ_levels
    n_regimes = length(σ_regimes)
    agent_belief = agent.regime_belief
    market_belief = perfect_regime_belief(vs)

    # 2. Look up actions in config from action struct
    half_spread    = config.spread_levels[action.spread_idx]
    hedge_fraction = config.hedge_targets[action.hedge_idx]

    # 3. Compute fair values for quoting and fills (only one option for now)
    opt        = env.current_options[1]
    V_believed = bs_all_belief_weighted(S, opt.K, τ, σ_regimes, agent_belief,  config.r; call=opt.is_call).price
    V_market   = bs_all_belief_weighted(S, opt.K, τ, σ_regimes, market_belief, config.r; call=opt.is_call).price

    # 4. Pre-action wealth
    port_before   = compute_portfolio(portfolio, env.current_options, S, τ, σ_regimes, agent_belief, config.r)
    wealth_before = port_before.portfolio_value + portfolio.cash

    # 5. Sim fills
    quotes = compute_quotes(V_believed, half_spread)
    fill   = simulate_fills(quotes.bid_price, quotes.ask_price, V_market, config, rng)

    # 6. Update inv and cash from fills
    update_from_fills!(portfolio, fill, 1)

    # 7. Recompute Δ_options after fills since Δ may have changed
    port_post_fills = compute_portfolio(
        portfolio, env.current_options, S, τ, σ_regimes, agent_belief, config.r
    )

    # 8. Execute hedge against Δ_options
    # return variables not needed for execution logic but will be used for visualization and evaluation
    shares_traded, hedge_cost = execute_hedge!(
        portfolio, hedge_fraction, port_post_fills.Δ_options, S, config
    )

    # 9. Step the spot market
    S_new, vs_new, log_return = step_spot(S, vs, config, rng)
    τ_new = τ - config.Δt

    # 10. Handle option expiry
    done = false
    if τ_new < config.Δt / 2
        reset_for_new_option!(portfolio, env.current_options, S_new)
        env.options_completed += 1
        τ_new = config.T_option * config.Δt  # reset τ for new option

        if env.options_completed >= config.n_options_per_episode
            done = true
        end
    end

    # 11. Update regime belief
    regime_belief_new = update_belief(
        agent_belief, vs_new, n_regimes, level,
        log_return, fill, env.current_options, S_new, τ_new, config,
        belief_update_fn
    )

    # 12. Post-action wealth
    port_after   = compute_portfolio(
        portfolio, env.current_options, S_new, τ_new, σ_regimes, regime_belief_new, config.r
    )
    wealth_after = port_after.portfolio_value + portfolio.cash

    # 13. Compute Reward
    # portfolio.q_spot is already post-hedge (execute_hedge! ran at step 8)
    # port_after.Δ_options is computed at new S_new, τ_new (post market move)
    net_Δ_after = port_after.Δ_options + portfolio.q_spot
    reward      = compute_reward(wealth_before, wealth_after, net_Δ_after, config)

    # 14. Build next agent state
    next_state = build_agent_state(
        portfolio, env.current_options, S_new, τ_new, regime_belief_new, σ_regimes, config.r
    )

    # 15 Update environment
    env.agent_state = next_state
    env.vol_state   = vs_new
    
    return next_state, reward, done, log_return, fill
end

