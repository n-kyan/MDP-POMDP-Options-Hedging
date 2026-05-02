include("1_types.jl")
include("2_black_scholes.jl")
include("3_spot_dynamics.jl")
include("4_fills.jl")
include("5_portfolio.jl")
include("12_belief_updater.jl")
include("6_environment.jl")

using Statistics: mean, std

# ============================================================
# Section 1: Spread Formulas (continuous output)
# ============================================================

# GLF-T (Guéant-Lehalle-Fernandez-Tapia 2013) optimal half-spread for options.
#
# Original derivation is for stocks; adapted by replacing σ²τ with the
# dollar-gamma term Γ·S²·σ²·τ (the relevant inventory risk for options).
#
# Formula: δ* = γ·|Γ|·S²·σ²·τ + (2/γ)·ln(1 + γ/k)
#   First term:  inventory risk — widen when gamma, vol, or remaining life is high
#   Second term: market-making edge from the spread itself (independent of inventory)
#
# Uses γ_market from config (the market maker's risk aversion in the GLF-T derivation).
function glft_half_spread(Γ::Float64, S::Float64, σ::Float64, τ::Float64, config::SimConfig)::Float64
    γ = config.γ_market
    k = config.k
    return γ * abs(Γ) * S^2 * σ^2 * τ + (2.0 / γ) * log(1.0 + γ / k)
end

# ============================================================
# Section 2: Hedge Formulas (continuous output)
# ============================================================

# Whalley-Wilmott (1997) no-trade band halfwidth.
#
# Formula: H = (3κ/(2φ) · Γ²·S²·σ²)^(1/3) · Δt^(1/3)
#
# Intuition: trade only when the cost of remaining unhedged exceeds the cost of a trade.
# Wide band (stay put) when γ is low or κ is high.
# Narrow band (hedge more) when γ is high or κ is low.
#
# Uses config.φ (agent's own risk aversion, consistent with the reward function).
function ww_band_halfwidth(Γ::Float64, S::Float64, σ::Float64, config::SimConfig)::Float64
    κ  = config.κ
    φ  = config.φ
    Δt = config.Δt
    dollar_gamma_sq = max(Γ^2 * S^2 * σ^2, 1e-10)
    return ((3κ / (2φ)) * dollar_gamma_sq)^(1/3) * Δt^(1/3)
end

# ============================================================
# Section 3: Action bound helpers
# ============================================================

# Lower bound of δ: cost to delta-hedge one option contract.
# Ensures spread at least covers the immediate hedging cost of a fill.
function δ_lower_bound(S::Float64, hat_Δ::Float64, config::SimConfig)::Float64
    return config.κ * S * abs(hat_Δ)
end

# Clamp a computed δ to the valid action range [lower_bound, hat_V].
function clamp_δ(δ::Float64, S::Float64, hat_Δ::Float64, hat_V::Float64, config::SimConfig)::Float64
    lo = δ_lower_bound(S, hat_Δ, config)
    hi = max(hat_V, lo + 1e-6)   # guard: upper must exceed lower
    return clamp(δ, lo, hi)
end

# ============================================================
# Section 4: Combined policies (emit continuous MarketMakingAction)
# ============================================================

# Primary benchmark: GLF-T spread + Whalley-Wilmott hedge.
#
# σ is passed in by the caller (typically oracle_σ for benchmarks).
# The policy reads hat_Δ_P, hat_Γ_P, hat_Δ, hat_V, S from env.agent_state.
function glft_ww_policy(
    env::EnvironmentState,
    portfolio::Portfolio,
    config::SimConfig,
    σ::Float64,
)::MarketMakingAction
    s = env.agent_state

    # GLF-T spread (continuous)
    δ_raw = glft_half_spread(s.hat_Γ_P, s.S, σ, s.τ, config)
    δ     = clamp_δ(δ_raw, s.S, s.hat_Δ, s.hat_V, config)

    # WW hedge target (continuous)
    H     = ww_band_halfwidth(s.hat_Γ_P, s.S, σ, config)
    Δ_target = if abs(s.hat_Δ_P) <= H
        s.hat_Δ_P   # inside band: no trade (target = current delta)
    else
        # Trade toward band edge; keep within action constraint [min(0,hat_Δ_P), max(0,hat_Δ_P)]
        band_edge = sign(s.hat_Δ_P) * H
        clamp(band_edge, min(0.0, s.hat_Δ_P), max(0.0, s.hat_Δ_P))
    end

    return MarketMakingAction(δ, Δ_target)
end

# GLF-T spread + naive full-hedge (always target net Δ = 0).
function glft_naive_policy(
    env::EnvironmentState,
    portfolio::Portfolio,
    config::SimConfig,
    σ::Float64,
)::MarketMakingAction
    s     = env.agent_state
    δ_raw = glft_half_spread(s.hat_Γ_P, s.S, σ, s.τ, config)
    δ     = clamp_δ(δ_raw, s.S, s.hat_Δ, s.hat_V, config)
    return MarketMakingAction(δ, 0.0)
end

# Fixed spread (0.10 half-spread or lower bound, whichever is larger) + WW hedge.
function naive_ww_policy(
    env::EnvironmentState,
    portfolio::Portfolio,
    config::SimConfig,
    σ::Float64,
)::MarketMakingAction
    s = env.agent_state
    δ = max(δ_lower_bound(s.S, s.hat_Δ, config), 0.10)

    H        = ww_band_halfwidth(s.hat_Γ_P, s.S, σ, config)
    Δ_target = if abs(s.hat_Δ_P) <= H
        s.hat_Δ_P
    else
        band_edge = sign(s.hat_Δ_P) * H
        clamp(band_edge, min(0.0, s.hat_Δ_P), max(0.0, s.hat_Δ_P))
    end

    return MarketMakingAction(δ, Δ_target)
end

# Fixed spread + naive full-hedge. Both components are maximally simple.
function naive_naive_policy(
    env::EnvironmentState,
    portfolio::Portfolio,
    config::SimConfig,
    σ::Float64,
)::MarketMakingAction
    s = env.agent_state
    δ = max(δ_lower_bound(s.S, s.hat_Δ, config), 0.10)
    return MarketMakingAction(δ, 0.0)
end

# ============================================================
# Section 5: Benchmark runner
# ============================================================

# σ_fn signature: (env::EnvironmentState) -> Float64
#   Oracle:    env -> sqrt(sum(perfect_regime_belief(env.vol_state) .* env.vol_state.vm.σ_levels .^ 2))
#   Particle:  env -> get_σ_hat(pf)   (uses the running particle filter)
function run_benchmark(
    policy_fn,
    σ_fn,
    vol_model::VolModel,
    config::SimConfig,
    n_episodes::Int,
    rng::AbstractRNG;
    n_particles::Int = 500,
    use_oracle::Bool = false,
)
    episode_pnl    = Vector{Float64}(undef, n_episodes)
    all_δ          = Float64[]
    all_hedged     = Bool[]
    all_abs_hat_Δ_P = Float64[]

    for ep in 1:n_episodes
        pf = ParticleFilter(n_particles)

        env = EnvironmentState(
            AgentState(NaN, 0.0, 0.0, 0, config.T_option * config.Δt,
                       get_σ_hat(pf), 0.0, 0.0, config.S0),
            VolState(vol_model),
            OptionContract[OptionContract(round(config.S0), true)],
            0
        )
        portfolio = Portfolio()
        push!(portfolio.option_quantities, 0)

        initialize_episode!(env, portfolio, vol_model, config, pf)

        if use_oracle
            ws_init = perfect_regime_belief(env.vol_state)
            bs_init = bs_all_belief_weighted(
                env.agent_state.S, env.current_options[1].K, env.agent_state.τ,
                env.vol_state.vm.σ_levels, ws_init, config.r; call = env.current_options[1].is_call
            )
            q0 = portfolio.option_quantities[1]
            env.agent_state = AgentState(
                NaN, q0 * bs_init.Δ + portfolio.q_spot, q0 * bs_init.Γ,
                0, env.agent_state.τ, sqrt(sum(ws_init .* env.vol_state.vm.σ_levels .^ 2)),
                bs_init.price, bs_init.Δ, env.agent_state.S
            )
        end

        ep_reward = 0.0
        done      = false

        while !done
            σ      = σ_fn(env)
            action = policy_fn(env, portfolio, config, σ)

            push!(all_δ,           action.δ)
            push!(all_hedged,      action.Δ_target != env.agent_state.hat_Δ_P)
            push!(all_abs_hat_Δ_P, abs(env.agent_state.hat_Δ_P))

            _, reward, done, _ = step_environment!(env, portfolio, pf, action, config, rng;
                                                   oracle_regime = use_oracle ? (env.vol_state.vm.σ_levels, perfect_regime_belief(env.vol_state)) : nothing)
            ep_reward += reward
        end

        episode_pnl[ep] = ep_reward
    end

    μ      = mean(episode_pnl)
    σ_pnl  = std(episode_pnl)
    sharpe = σ_pnl > 1e-10 ? μ / σ_pnl : 0.0

    return (
        episode_pnl      = episode_pnl,
        sharpe           = sharpe,
        mean_δ           = mean(all_δ),
        hedge_freq       = mean(all_hedged),
        mean_abs_hat_Δ_P = mean(all_abs_hat_Δ_P),
    )
end
