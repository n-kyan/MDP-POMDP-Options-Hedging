# ============================================================
# Module 7: Analytical Benchmarks
# 7_benchmarks.jl
# ============================================================
# Provides two analytical benchmark policies to compare against the RL agent:
#
#   1. glft_ww_policy  — GLF-T spread + Whalley-Wilmott hedge (primary benchmark)
#   2. symmetric_naive_policy — fixed spread + naive BS delta hedge (sanity floor)
#
# Each policy is run in two modes controlled by the σ_fn parameter:
#   - Oracle:        σ_fn = env -> get_σ(env.vol_state)   (cheats, uses true regime vol)
#   - Constant-vol:  σ_fn = env -> 0.20                   (fixed sigma, no regime info)
#
# All benchmark formulas take σ as an explicit argument — the benchmarks themselves
# are agnostic about where σ comes from. The caller decides via σ_fn.
# ============================================================

include("1_types.jl")
include("2_black_scholes.jl")
include("3_spot_dynamics.jl")
include("4_fills.jl")
include("5_portfolio.jl")
include("6_environment.jl")

using Statistics: mean, std

# ============================================================
# Section 1: GLF-T Spread Formula
# ============================================================

#=
GLF-T (Guéant-Lehalle-Fernandez-Tapia 2013) optimal half-spread, adapted for options.

Original GLF-T derivation is for stocks, where inventory risk scales with σ²τ.
For options, the relevant risk is dollar gamma: the P&L of an options position
varies as (1/2)·Γ·S²·(dS)², so the inventory risk term becomes Γ·S²·σ²·τ.

Formula: δ* = γ·|Γ|·S²·σ²·τ + (2/γ)·ln(1 + γ/k)
  - First term:  inventory risk component — widen when gamma is high, vol is high, expiry is far
  - Second term: market-making component — earn positive edge from the spread itself

Note: We use |Γ| because net_Γ can be negative for a short-gamma book.
The spread should widen with gamma exposure regardless of sign.

Args:
  Γ      — net portfolio gamma (from agent_state.net_Γ)
  S      — current spot price
  σ      — volatility to use (caller decides: true regime vol or constant)
  τ      — time to expiry in years
  config — SimConfig (uses φ as γ and k as fill decay rate)

Returns: continuous dollar half-spread (positive Float64)
=#
function glft_half_spread(Γ::Float64, S::Float64, σ::Float64, τ::Float64, config::SimConfig)::Float64
    γ = config.φ
    k = config.k
    inventory_term     = γ * abs(Γ) * S^2 * σ^2 * τ
    market_making_term = (2.0 / γ) * log(1.0 + γ / k)
    return inventory_term + market_making_term
end

# Snap a continuous half-spread to the nearest discrete spread level index.
# Example: half_spread=0.13, spread_levels=[0.05,0.10,0.20,0.40,0.80]
#   diffs = [0.08, 0.03, 0.07, 0.27, 0.67] → argmin = 2 (the 0.10 level)
function nearest_spread_idx(half_spread::Float64, config::SimConfig)::Int
    diffs = abs.(config.spread_levels .- half_spread)
    return argmin(diffs)
end

# GLF-T spread policy: compute optimal half-spread from current Greeks, snap to grid.
function glft_spread_idx(env::EnvironmentState, config::SimConfig, σ::Float64)::Int
    s  = env.agent_state
    hs = glft_half_spread(s.net_Γ, s.S, σ, s.τ, config)
    return nearest_spread_idx(hs, config)
end

# Symmetric (naive) policy: always return the same fixed spread level index.
# Ignores all market conditions. Default is level 2 ($0.10 half-spread).
function symmetric_spread_idx(; spread_level_idx::Int = 2)::Int
    return spread_level_idx
end

# ============================================================
# Section 2: Whalley-Wilmott Hedge Formula
# ============================================================

#=
Whalley-Wilmott (1997) no-trade band halfwidth.

WW solves the problem: given transaction costs κ, when is it optimal to rebalance?
The answer is: only when your net-delta drifts outside a band of width ±H around zero.
Inside the band, the cost of rebalancing exceeds the expected benefit. Outside, trade
back to the band edge (not all the way to zero — that would overshoot).

Formula: H = (3κ/(2φ) · Γ²·S²·σ²)^(1/3) · Δt^(1/3)

The cubic root comes from balancing rebalancing cost (linear in trade size) against
the expected drift cost (quadratic in delta). The Δt^(1/3) scaling means with more
frequent rebalancing steps, each individual band can be tighter.

Intuition for band width:
  - Wide band (stay put): low gamma (delta moves slowly), high κ (expensive to trade)
  - Narrow band (trade often): high gamma (delta moves fast), low κ (cheap to trade)

Returns: half-width H in net-delta units
=#
function ww_band_halfwidth(Γ::Float64, S::Float64, σ::Float64, config::SimConfig)::Float64
    κ  = config.κ
    φ  = config.φ
    Δt = config.Δt
    # Dollar-gamma-squared: how fast the delta risk accumulates
    dollar_gamma_sq = Γ^2 * S^2 * σ^2
    dollar_gamma_sq = max(dollar_gamma_sq, 1e-10)  # guard against zero gamma
    H = ((3κ / (2φ)) * dollar_gamma_sq)^(1/3) * Δt^(1/3)
    return H
end

#=
Whalley-Wilmott hedge policy.

Logic:
  1. Compute no-trade band halfwidth H
  2. If |net_Δ| ≤ H → no_trade (inside the band, don't pay transaction costs)
  3. If |net_Δ| > H → trade toward the band edge, not all the way to zero
     Band edge is at sign(net_Δ) × H. We snap to the nearest Δ_target from there.
=#
function ww_hedge_idx(env::EnvironmentState, config::SimConfig, σ::Float64)::Int
    s     = env.agent_state
    net_Δ = s.net_Δ
    H     = ww_band_halfwidth(s.net_Γ, s.S, σ, config)

    if abs(net_Δ) <= H
        return 1  # index 1 is always :no_trade by convention
    else
        # Target the band edge: sign tells us direction, H tells us how far
        band_edge = sign(net_Δ) * H
        # Clamp to our grid range — Δ_targets[2] is the most negative numeric target
        # (e.g. -0.3), Δ_targets[end] is the most positive (e.g. +0.3)
        numeric_targets = Float64.(config.Δ_targets[2:end])
        target = clamp(band_edge, minimum(numeric_targets), maximum(numeric_targets))
        # Find nearest numeric Δ_target (skip index 1 which is :no_trade)
        diffs = abs.(numeric_targets .- target)
        return argmin(diffs) + 1  # +1 because we skipped index 1
    end
end

# Naive delta hedge: always target net_Δ = 0.0, every step, no transaction cost awareness.
# This is the textbook Black-Scholes hedge — rebalance fully every step.
function naive_hedge_idx(config::SimConfig)::Int
    numeric_targets = Float64.(config.Δ_targets[2:end])
    diffs = abs.(numeric_targets .- 0.0)
    return argmin(diffs) + 1  # +1 to skip :no_trade at index 1
end

# ============================================================
# Section 3: Combined Policies → MarketMakingAction
# ============================================================

# Primary benchmark: GLF-T spread + Whalley-Wilmott hedge.
# Pass σ_fn result as σ. For oracle: σ = get_σ(env.vol_state). For constant: σ = 0.20.
function glft_ww_policy(
    env::EnvironmentState,
    portfolio::Portfolio,
    config::SimConfig,
    σ::Float64
)::MarketMakingAction
    return MarketMakingAction(
        glft_spread_idx(env, config, σ),
        ww_hedge_idx(env, config, σ)
    )
end

# Sanity floor: fixed spread + naive BS delta hedge.
function symmetric_naive_policy(
    env::EnvironmentState,
    portfolio::Portfolio,
    config::SimConfig,
    σ::Float64;
    spread_level_idx::Int = 2
)::MarketMakingAction
    return MarketMakingAction(
        symmetric_spread_idx(; spread_level_idx = spread_level_idx),
        naive_hedge_idx(config)
    )
end

# ============================================================
# Section 4: Benchmark Runner
# ============================================================

#=
Run a benchmark policy for n_episodes and collect per-step and per-episode statistics.

Args:
  policy_fn  — function (env, portfolio, config, σ) → MarketMakingAction
  σ_fn       — function (env) → Float64
               Oracle:       env -> get_σ(env.vol_state)   uses true regime vol
               Constant-vol: env -> 0.20                   ignores regime
  vol_model  — VolModel used to initialize episodes
  config     — SimConfig
  n_episodes — number of Monte Carlo episodes to run
  rng        — seeded AbstractRNG for reproducibility
  level      — simulation level (1 = constant vol, 2 = known regime, 3 = hidden)

Returns a NamedTuple:
  episode_pnl     — Vector{Float64}: total reward per episode
  sharpe          — Float64: mean(episode_pnl) / std(episode_pnl)
  mean_spread_idx — Float64: average spread level index chosen across all steps
  hedge_freq      — Float64: fraction of steps where a hedge trade occurred (not :no_trade)
  mean_abs_net_Δ  — Float64: average |net_Δ| across all steps (hedging quality)
=#
function run_benchmark(
    policy_fn,
    σ_fn,
    vol_model::VolModel,
    config::SimConfig,
    n_episodes::Int,
    rng::AbstractRNG;
    level::Int = 1
)
    episode_pnl    = Vector{Float64}(undef, n_episodes)
    all_spread_idx = Float64[]
    all_hedged     = Bool[]
    all_net_Δ      = Float64[]

    for ep in 1:n_episodes
        # Initialize a fresh episode
        env       = EnvironmentState(
            AgentState(
                config.S0,
                config.T_option * config.Δt,
                0.0, 0.0, 0.0, 0.0,
                fill(1.0 / length(vol_model.σ_levels), length(vol_model.σ_levels))
            ),
            VolState(vol_model),
            OptionContract[OptionContract(round(config.S0), true)],
            0
        )
        portfolio = Portfolio()
        push!(portfolio.option_quantities, 0)

        initialize_episode!(env, portfolio, vol_model, config; level = level)

        ep_reward = 0.0
        done      = false

        while !done
            σ      = σ_fn(env)
            action = policy_fn(env, portfolio, config, σ)

            push!(all_spread_idx, Float64(action.spread_idx))
            push!(all_hedged,     config.Δ_targets[action.hedge_idx] != :no_trade)
            push!(all_net_Δ,      abs(env.agent_state.net_Δ))

            _, reward, done, _, _ = step_environment!(
                env, portfolio, action, config, rng; level = level
            )
            ep_reward += reward
        end

        episode_pnl[ep] = ep_reward
    end

    μ     = mean(episode_pnl)
    σ_pnl = std(episode_pnl)
    sharpe = σ_pnl > 1e-10 ? μ / σ_pnl : 0.0

    return (
        episode_pnl     = episode_pnl,
        sharpe          = sharpe,
        mean_spread_idx = mean(all_spread_idx),
        hedge_freq      = mean(all_hedged),
        mean_abs_net_Δ  = mean(all_net_Δ)
    )
end