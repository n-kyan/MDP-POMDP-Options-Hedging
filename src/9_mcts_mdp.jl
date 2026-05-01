include("7_benchmarks.jl")

using POMDPs
using MCTS
using POMDPTools
using StatsBase: Weights, sample

# ============================================================
# Section 1: State and MDP definition
# ============================================================

# Full state for the oracle MDP. regime_idx is observable (oracle).
# opt_K tracks the current option strike, which resets at each rollover.
struct MDPState
    S::Float64
    regime_idx::Int
    q_options::Int
    opt_K::Float64
    q_spot::Float64
    cash::Float64
    τ::Float64
    options_completed::Int
end

struct OptionsMM_MDP <: MDP{MDPState, MarketMakingAction}
    config::SimConfig
    vm::VolModel
end

# ============================================================
# Section 2: POMDPs.jl interface — MDP
# ============================================================

POMDPs.discount(m::OptionsMM_MDP) = 1.0

POMDPs.isterminal(m::OptionsMM_MDP, s::MDPState) =
    s.options_completed >= m.config.n_options_per_episode

function POMDPs.initialstate(m::OptionsMM_MDP)
    config   = m.config
    regime   = sample(1:length(m.vm.σ_levels), Weights(m.vm.stationary_dist))
    s0 = MDPState(
        config.S0,
        regime,
        0,
        Float64(round(config.S0)),
        0.0,
        0.0,
        config.T_option * config.Δt,
        0,
    )
    return POMDPTools.Deterministic(s0)
end

# Pure generative model: true Hardy dynamics, oracle pricing (hat_V = V_market).
function POMDPs.gen(m::OptionsMM_MDP, s::MDPState, a::MarketMakingAction, rng::AbstractRNG)
    config = m.config
    vm     = m.vm
    r      = config.r
    Δt     = config.Δt

    # Transition-row belief and oracle vol
    ws        = vm.transition_matrix[s.regime_idx, :]
    oracle_σ  = sqrt(sum(ws .* vm.σ_levels .^ 2))

    # True market price (exact belief-weighted BS, not Jensen approximation)
    V_market = bs_all_belief_weighted(
        s.S, s.opt_K, s.τ, vm.σ_levels, ws, r; call = true
    ).price

    # Oracle agent: hat_V = V_market (no mispricing)
    hat_V     = V_market
    bid_price = hat_V - a.δ
    ask_price = hat_V + a.δ

    # Oracle BS Greeks (for delta computation)
    bs = bs_all(s.S, s.opt_K, s.τ, oracle_σ, r; call = true)

    # Wealth before (marked at true V_market)
    wealth_before = s.cash + s.q_options * V_market + s.q_spot * s.S

    # Fills against V_market
    fill = simulate_fills(bid_price, ask_price, V_market, config, rng)

    q_new    = s.q_options + (fill.bid_filled ? 1 : 0) - (fill.ask_filled ? 1 : 0)
    cash_new = s.cash
    if fill.bid_filled; cash_new -= bid_price; end
    if fill.ask_filled; cash_new += ask_price; end

    # Hedge: bring net delta from post-fill position toward Δ_target
    hat_Δ_P_post = q_new * bs.Δ + s.q_spot
    shares       = a.Δ_target - hat_Δ_P_post
    hedge_cost   = config.κ * abs(shares) * s.S
    cash_new    -= shares * s.S + hedge_cost
    q_spot_new   = s.q_spot + shares

    # Spot step: GBM with oracle σ
    Z     = randn(rng)
    S_new = s.S * exp((r - 0.5 * oracle_σ^2) * Δt + oracle_σ * sqrt(Δt) * Z)
    τ_new = s.τ - Δt

    # Regime transition (true Hardy dynamics)
    new_regime = sample(rng, 1:length(vm.σ_levels), Weights(ws))

    # Option expiry: settle at true contractual payoff (vol-free)
    q_opt_new   = q_new
    opt_K_new   = s.opt_K
    opts_done   = s.options_completed
    if τ_new < Δt / 2
        cash_new  += q_opt_new * max(S_new - s.opt_K, 0.0)
        q_opt_new  = 0
        opt_K_new  = Float64(round(S_new))
        opts_done += 1
        τ_new      = config.T_option * Δt
    end

    # Wealth after (new regime's oracle V_market)
    ws_new = vm.transition_matrix[new_regime, :]
    if q_opt_new > 0
        V_market_new = bs_all_belief_weighted(
            S_new, opt_K_new, τ_new, vm.σ_levels, ws_new, r; call = true
        ).price
    else
        V_market_new = 0.0
    end
    wealth_after = cash_new + q_opt_new * V_market_new + q_spot_new * S_new

    reward = (wealth_after - wealth_before) - config.φ * a.Δ_target^2

    sp = MDPState(S_new, new_regime, q_opt_new, opt_K_new, q_spot_new, cash_new, τ_new, opts_done)
    return (sp = sp, r = reward)
end

# ============================================================
# Section 3: Action widening for MCTS+DPW
# ============================================================

struct OracleMDPActionSampler
    config::SimConfig
    vm::VolModel
end

function MCTS.next_action(sampler::OracleMDPActionSampler, m::OptionsMM_MDP, s::MDPState, h)
    config    = sampler.config
    vm        = sampler.vm
    ws        = vm.transition_matrix[s.regime_idx, :]
    oracle_σ  = sqrt(sum(ws .* vm.σ_levels .^ 2))
    bs        = bs_all(s.S, s.opt_K, s.τ, oracle_σ, config.r; call = true)

    hat_Δ_P = s.q_options * bs.Δ + s.q_spot

    δ_lo     = δ_lower_bound(s.S, bs.Δ, config)
    δ_hi     = max(bs.price, δ_lo + 1e-6)
    δ        = δ_lo + rand() * (δ_hi - δ_lo)

    Δ_lo     = min(0.0, hat_Δ_P)
    Δ_hi     = max(0.0, hat_Δ_P)
    Δ_target = Δ_lo + rand() * (Δ_hi - Δ_lo + 1e-10)

    return MarketMakingAction(δ, Δ_target)
end

# ============================================================
# Section 4: Rollout policy — oracle GLF-T+WW
# ============================================================

struct OracleGLFTRollout <: Policy
    config::SimConfig
    vm::VolModel
end

function POMDPs.action(p::OracleGLFTRollout, s::MDPState)
    config   = p.config
    vm       = p.vm
    ws       = vm.transition_matrix[s.regime_idx, :]
    oracle_σ = sqrt(sum(ws .* vm.σ_levels .^ 2))
    bs       = bs_all(s.S, s.opt_K, s.τ, oracle_σ, config.r; call = true)

    hat_Δ_P = s.q_options * bs.Δ + s.q_spot
    hat_Γ_P = s.q_options * bs.Γ

    δ_raw    = glft_half_spread(hat_Γ_P, s.S, oracle_σ, s.τ, config)
    δ        = clamp_δ(δ_raw, s.S, bs.Δ, bs.price, config)

    H        = ww_band_halfwidth(hat_Γ_P, s.S, oracle_σ, config)
    Δ_target = if abs(hat_Δ_P) <= H
        hat_Δ_P
    else
        band_edge = sign(hat_Δ_P) * H
        clamp(band_edge, min(0.0, hat_Δ_P), max(0.0, hat_Δ_P))
    end

    return MarketMakingAction(δ, Δ_target)
end

# ============================================================
# Section 5: Solver factory
# ============================================================

function make_mcts_solver(config::SimConfig, vm::VolModel;
                          n_queries::Int = 50,
                          max_depth::Int = 5,
                          exploration_constant::Float64 = 1.0,
                          k_action::Float64 = 4.0,
                          alpha_action::Float64 = 0.5,
                          seed::Int = 42)
    return DPWSolver(
        n_iterations         = n_queries,
        depth                = max_depth,
        exploration_constant = exploration_constant,
        k_action             = k_action,
        alpha_action         = alpha_action,
        next_action          = OracleMDPActionSampler(config, vm),
        estimate_value       = MCTS.RolloutEstimator(OracleGLFTRollout(config, vm)),
        rng                  = MersenneTwister(seed),
        keep_tree            = false,
    )
end

# ============================================================
# Section 6: Evaluation
# ============================================================

function evaluate_mcts_mdp(
    vm::VolModel,
    config::SimConfig,
    n_episodes::Int,
    seed::Int;
    n_queries::Int = 50,
    max_depth::Int = 5,
)
    mdp     = OptionsMM_MDP(config, vm)
    solver  = make_mcts_solver(config, vm; n_queries, max_depth, seed)
    rng_env = MersenneTwister(seed + 1)

    episode_rewards = Float64[]
    pf_dummy = ParticleFilter(10)   # satisfies step_environment! API; σ_hat_override bypasses it

    for ep in 1:n_episodes
        # --- True Hardy environment ---
        vs       = VolState(vm)
        env      = EnvironmentState(
            AgentState(NaN, 0.0, 0.0, 0, config.T_option * config.Δt, 0.2, 0.0, 0.0, config.S0),
            vs,
            OptionContract[OptionContract(Float64(round(config.S0)), true)],
            0,
        )
        portfolio = Portfolio()
        push!(portfolio.option_quantities, 0)
        initialize_episode!(env, portfolio, vm, config, pf_dummy)

        # Sync MDP state with true environment starting regime
        mdp_state = MDPState(
            config.S0,
            env.vol_state.regime_idx,
            0,
            Float64(round(config.S0)),
            0.0,
            0.0,
            config.T_option * config.Δt,
            0,
        )

        ep_reward = 0.0
        done      = false

        while !done
            # Replan at current MDP state
            planner = solve(solver, mdp)
            action  = POMDPs.action(planner, mdp_state)

            # Oracle σ for the true environment step
            ws_cur     = vm.transition_matrix[env.vol_state.regime_idx, :]
            σ_override = sqrt(sum(ws_cur .* vm.σ_levels .^ 2))

            # Execute in TRUE environment (Hardy dynamics)
            _, reward, done, _ = step_environment!(
                env, portfolio, pf_dummy, action, config, rng_env;
                σ_hat_override = σ_override,
            )
            ep_reward += reward

            # Sync MDP state from true environment (oracle has perfect state knowledge)
            mdp_state = MDPState(
                env.agent_state.S,
                env.vol_state.regime_idx,
                portfolio.option_quantities[1],
                env.current_options[1].K,
                portfolio.q_spot,
                portfolio.cash,
                env.agent_state.τ,
                env.options_completed,
            )
        end

        push!(episode_rewards, ep_reward)
    end

    μ      = Statistics.mean(episode_rewards)
    σ_pnl  = Statistics.std(episode_rewards)
    sharpe = σ_pnl > 1e-10 ? μ / σ_pnl : 0.0

    return (episode_rewards = episode_rewards, mean_reward = μ, std_reward = σ_pnl, sharpe = sharpe)
end
