#=
============================================================================
Module 6: Environment Step Function — environment.jl
============================================================================

Wires modules 2–5 into a single `step_environment!` function and provides
`initialize_episode!` to reset the simulation at the start of each episode.

Architectural role:
  - All pure math lives in black_scholes.jl and fills.jl.
  - All accounting lives in portfolio.jl.
  - This module is ONLY orchestration: call the right functions in the right
    order, pass results between them, and manage state transitions.

Belief update design:
  The `belief_update_fn` keyword argument makes the belief update pluggable:
    nothing (default) → Level 1/2: deterministic update (see below)
    a function         → Level 3: Hamilton filter (provided by belief_updater.jl)

  Level 1 (n_regimes == 1): belief = [1.0], always.
  Level 2 (n_regimes > 1, no fn provided): agent has perfect observability,
    belief is one-hot on the true new regime after the market step.
  Level 3 (fn provided): fn(belief, log_return, fill, ...) → new belief.
============================================================================
=#

# ─────────────────────────────────────────────────────────────────────────────
# Episode Initialization
# ─────────────────────────────────────────────────────────────────────────────

#=
Reset all simulation state to start a new episode.

Sets up:
  - A fresh Portfolio (zero inventory, zero cash, zero spot)
  - A new VolState sampled from the stationary distribution
  - An initial ATM call option at K = round(S0)
  - Initial regime belief (Level 1/2 deterministic; Level 3 = stationary dist)
  - Initial AgentState at τ = T_option × Δt years

Returns the initial AgentState so the agent can start selecting actions.
=#
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

    # Fresh vol state (regime sampled from stationary distribution)
    env.vol_state = VolState(vm)
    env.options_completed = 0

    # Initial ATM call at S0
    empty!(env.current_options)
    push!(env.current_options, OptionContract(round(config.S0), true))
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
        # Perfect observability: one-hot on the true starting regime
        return [vs.regime_idx == i ? 1.0 : 0.0 for i in 1:n_regimes]
    else
        # Level 3: start from stationary distribution (agent doesn't know initial regime)
        return copy(vs.vm.stationary_dist)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Main Step Function
# ─────────────────────────────────────────────────────────────────────────────

#=
Advance the environment by one timestep given the agent's chosen action.

Inputs:
  env            — current EnvironmentState (holds AgentState, VolState, options)
  portfolio      — ground-truth Portfolio (inventory, spot, cash)
  action         — MarketMakingAction chosen by the agent for this step
  config         — SimConfig (parameters, action space)
  rng            — random number generator (pass for reproducibility)
  belief_update_fn — optional: Hamilton filter for Level 3. Signature:
                     (belief, log_return, fill, options, S, config) → new_belief
                     Pass `nothing` for Levels 1 and 2.

Returns:
  next_state   — AgentState at t+1 (what the agent observes next)
  reward       — scalar reward r_t = ΔWealth - φ·net_Δ²
  done         — true if the episode is complete
  log_return   — log(S_new/S): returned for logging and Level 3 belief update
  fill         — FillOutcome: returned for logging and Level 3 belief update
=#
function step_environment!(
    env::EnvironmentState,
    portfolio::Portfolio,
    action::MarketMakingAction,
    config::SimConfig,
    rng::AbstractRNG;
    belief_update_fn = nothing,
    level::Int = 2
)
    # ── 1. Unpack current state ──────────────────────────────────────────────
    ag        = env.agent_state
    vs        = env.vol_state
    S         = ag.S
    τ         = ag.τ
    vm        = vs.vm
    σ_regimes = vm.σ_levels
    n_regimes = length(σ_regimes)

    agent_belief  = ag.regime_belief
    market_belief = market_regime_belief(vs)  # true transition row from current regime

    # ── 2. Look up action parameters ────────────────────────────────────────
    half_spread    = config.spread_levels[action.spread_idx]
    hedge_fraction = config.hedge_targets[action.hedge_idx]

    # ── 3. Compute fair values for quoting and fills ─────────────────────────
    #
    # V_believed: agent's best estimate of fair value, used to center its quotes.
    #   Computed from agent's regime_belief — may differ from V_market in Level 3.
    #
    # V_market: market's consensus price, used to determine fill probabilities.
    #   Computed from the true transition row (perfect knowledge of current regime).
    #
    # In Levels 1-2, agent_belief == market_belief, so V_believed == V_market
    # and fills are symmetric. In Level 3 after a regime switch, these diverge,
    # creating asymmetric fills that signal the agent its pricing is off.
    #
    # NOTE: Currently quotes only one option (index 1). Multi-strike extension
    # would loop over env.current_options here, quoting each separately.
    opt       = env.current_options[1]
    V_believed = bs_all_belief_weighted(
        S, opt.K, τ, σ_regimes, agent_belief, config.r; call=opt.is_call
    ).price
    V_market = bs_all_belief_weighted(
        S, opt.K, τ, σ_regimes, market_belief, config.r; call=opt.is_call
    ).price

    # ── 4. Snapshot wealth before any action ────────────────────────────────
    #
    # Wealth = mark-to-market portfolio value + cash.
    # Snapshotted NOW, before fills and hedge change inventory or cash.
    # The reward is computed as wealth_after - wealth_before at the end.
    port_before   = compute_portfolio(portfolio, env.current_options, S, τ, σ_regimes, agent_belief, config.r)
    wealth_before = port_before.portfolio_value + portfolio.cash

    # ── 5. Simulate fills ────────────────────────────────────────────────────
    quotes = compute_quotes(V_believed, half_spread)
    fill   = simulate_fills(quotes.bid_price, quotes.ask_price, V_market, config, rng)

    # ── 6. Update inventory and cash from fills ──────────────────────────────
    update_from_fills!(portfolio, fill, 1)

    # ── 7. Recompute Δ_options POST-fills ────────────────────────────────────
    #
    # Fills changed option_quantities, so Δ_options has changed.
    # The hedge must target the CURRENT delta exposure, not pre-fill delta.
    # This is the correct behavior: the agent hedges what it actually holds.
    port_post_fills = compute_portfolio(
        portfolio, env.current_options, S, τ, σ_regimes, agent_belief, config.r
    )

    # ── 8. Execute hedge against updated Δ_options ──────────────────────────
    shares_traded, hedge_cost = execute_hedge!(
        portfolio, hedge_fraction, port_post_fills.Δ_options, S, config
    )

    # ── 9. Step the market ───────────────────────────────────────────────────
    #
    # The market moves AFTER the agent has quoted and hedged.
    # This move generates the bulk of the reward: option and spot values
    # change as S moves, and the agent's hedge quality determines the P&L.
    S_new, vs_new, log_return = step_spot(S, vs, config, rng)
    τ_new = τ - config.Δt

    # ── 10. Handle option expiry ─────────────────────────────────────────────
    #
    # Use a small tolerance rather than exact zero comparison to handle
    # floating point drift from repeated subtraction of 1/252.
    done = false
    if τ_new < config.Δt / 2
        reset_for_new_option!(portfolio, env.current_options, S_new)
        env.options_completed += 1
        τ_new = config.T_option * config.Δt  # reset τ for new option

        if env.options_completed >= config.n_options_per_episode
            done = true
        end
    end

    # ── 11. Update regime belief ─────────────────────────────────────────────
    regime_belief_new = update_belief(
        agent_belief, vs_new, n_regimes, level,
        log_return, fill, env.current_options, S_new, τ_new, config,
        belief_update_fn
    )

    # ── 12. Snapshot wealth after ────────────────────────────────────────────
    #
    # Use new S, new τ, new belief, and updated portfolio (post-fills, post-hedge).
    # The difference wealth_after - wealth_before captures:
    #   + spread income from fills (via cash changes in update_from_fills!)
    #   + option mark-to-market change as S moved (via portfolio_value change)
    #   + spot hedge mark-to-market change as S moved (via portfolio_value change)
    #   - hedge transaction costs (via cash deduction in execute_hedge!)
    port_after   = compute_portfolio(
        portfolio, env.current_options, S_new, τ_new, σ_regimes, regime_belief_new, config.r
    )
    wealth_after = port_after.portfolio_value + portfolio.cash

    # ── 13. Compute reward ───────────────────────────────────────────────────
    net_Δ_after = port_after.Δ_options + portfolio.q_spot
    reward      = compute_reward(wealth_before, wealth_after, net_Δ_after, config)

    # ── 14. Build next agent state ───────────────────────────────────────────
    next_state = build_agent_state(
        portfolio, env.current_options, S_new, τ_new, regime_belief_new, σ_regimes, config.r
    )

    # ── 15. Update environment ───────────────────────────────────────────────
    env.agent_state = next_state
    env.vol_state   = vs_new

    return next_state, reward, done, log_return, fill
end

# ─────────────────────────────────────────────────────────────────────────────
# Belief Update Router
# ─────────────────────────────────────────────────────────────────────────────

#=
Routes belief update to the correct logic based on simulation level.

Level 1 (n_regimes == 1 or level == 1):
  Belief is always [1.0]. No update needed.

Level 2 (no belief_update_fn provided):
  Agent has perfect observability. After each market step, the agent
  knows the true new regime → one-hot encoding of vs_new.regime_idx.

Level 3 (belief_update_fn provided):
  Hamilton filter from belief_updater.jl. The function receives the prior
  belief, the log return, and the fill outcome, and returns a posterior.
  Implemented in Module 12.

This router is a separate function so step_environment! stays readable.
=#
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