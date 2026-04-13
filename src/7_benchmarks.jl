include("1_types.jl")
include("2_black_scholes.jl")
include("3_spot_dynamics.jl")
include("4_fills.jl")
include("5_portfolio.jl")
include("6_environment.jl")

using Statistics: mean, std

# ============================================================
# Section 1: Spread Benchmark Formulas (pure math, no state)
# ============================================================

#=
AS (Avellaneda-Stoikov) optimal half-spread, adapted for options.
Original AS was for stocks: δ* = (1/k)ln(1+γ/k) + γσ²τ
Options adaptation: replace σ²τ with Γ·S²·σ²·τ (dollar gamma term)
Returns: continuous dollar half-spread (not yet snapped to discrete levels)
=#
function as_half_spread(
    Γ::Float64, S::Float64, σ::Float64, τ::Float64, config::SimConfig
    )::Float64
    γ = config.γ_market
    k = config.k
    inventory_term = γ * abs(Γ) * S^2 * σ^2 * τ
    market_making_term = (1.0 / k) * log(1.0 + γ / k)
    return market_making_term + inventory_term
end

#=
GLF-T (Guéant-Lehalle-Fernandez-Tapia) optimal half-spread, adapted for options.
Same components as AS but derived from a different (more rigorous) control problem.
Has a full proof of optimality under exponential utility — preferred over AS as primary benchmark.
Formula: δ* = γ·Γ·S²·σ²·τ + (2/γ)·ln(1 + γ/k)
=#
function glft_half_spread(
    Γ::Float64, S::Float64, σ::Float64, τ::Float64, config::SimConfig
    )::Float64
    γ = config.γ_market
    k = config.k
    inventory_term = γ * abs(Γ) * S^2 * σ^2 * τ
    market_making_term = (2.0 / γ) * log(1.0 + γ / k)
    return inventory_term + market_making_term
end

# ============================================================
# Section 2: Spread Policies 
# ============================================================

#=
Snap a continuous half-spread (in dollars) to the nearest discrete spread level index.
Uses argmin over abs(spread_levels .- half_spread).
Clamps to valid index range [1, n_levels].
=#
function nearest_spread_idx(half_spread::Float64, config::SimConfig)::Int
    diffs = abs.(config.spread_levels .- half_spread)
    return argmin(diffs)
end

# AS spread policy: compute AS optimal half-spread for current Greeks, snap to nearest level.
# Reads Γ, S, τ from env.agent_state.
function as_spread_idx(env::EnvironmentState, config::SimConfig, σ::Float64)::Int
    s = env.agent_state
    hs = as_half_spread(s.net_Γ, s.S, σ, s.τ, config)
    return nearest_spread_idx(hs, config)
end

# GLF-T spread policy: compute GLF-T optimal half-spread, snap to nearest level.
function glft_spread_idx(env::EnvironmentState, config::SimConfig, σ::Float64)::Int
    s = env.agent_state
    hs = glft_half_spread(s.net_Γ, s.S, σ, s.τ, config)
    return nearest_spread_idx(hs, config)
end

# Symmetric (naive) policy: always return the same fixed spread level index.
# Caller passes which level (e.g. 2 for the $0.10 half-spread).
# Ignores all market conditions — baseline sanity check.
function symmetric_spread_idx(spread_level_idx::Int = 2)::Int
    return spread_level_idx
end

# ============================================================
# Section 3: Hedge Benchmark Formulas (pure math, no state)
# ============================================================

#=
Whalley-Wilmott no-trade band halfwidth.
Formula: H = (3κ/(2φ) · Γ²·S²·σ²)^(1/3) · Δt^(1/3)

Intuition: The band represents the range of net-delta where the cost of rebalancing exceeds the expected benefit. Inside the band, stay put. Outside, trade back to the edge.

 Wider band (don't trade) when:
   - gamma is LOW (delta moves slowly, can afford to wait)
   - transaction costs κ are HIGH (rebalancing is expensive)
   - risk aversion φ is LOW (you don't care as much about delta risk)

 Narrower band (trade more often) when:
   - gamma is HIGH (delta moves fast, need to track closely)
   - transaction costs κ are LOW (cheap to rebalance)
   - risk aversion φ is HIGH (you hate unhedged exposure)

Returns: half-width in net-delta units (not dollars)
=#

function ww_band_halfwidth(Γ::Float64, S::Float64, σ::Float64, config::SimConfig)::Float64
    κ = config.κ
    φ = config.γ_market
    Δt = config.Δt
    dollar_gamma_sq = Γ^2 * S^2 * σ^2
    # Guard against zero gamma (ATM call at inception can have small but nonzero Γ)
    dollar_gamma_sq = max(dollar_gamma_sq, 1e-10)
    H = ((3κ / (2φ)) * dollar_gamma_sq)^(1/3) * Δt^(1/3)
    return H
end

function leland_modified_vol(σ::Float64, config::SimConfig)::Float64
    κ = config.κ
    Δt = config.Δt
    inflation = sqrt(2.0 / π) * κ / (σ * sqrt(Δt))
    σ̂ = sqrt(σ^2 * (1.0 + inflation))
    return σ̂
end

# ============================================================
# Section 4: Hedge Policies (formula → discrete hedge_idx)
# ============================================================

function no_trade_idx()::Int
    return 1
end

function nearest_hedge_idx(target_Δ::Float64, config::SimConfig)::Int
    best_idx = 2  # start from index 2 (first numeric target)
    best_dist = Inf
    for i in 2:length(config.Δ_targets)
        d = abs(config.Δ_targets[i] - target_Δ)
        if d < best_dist
            best_dist = d
            best_idx = i
        end
    end
    return best_idx
end

function ww_hedge_idx(env::EnvironmentState, config::SimConfig, σ::Float64)::Int
    s = env.agent_state
    net_Δ = s.net_Δ
    H = ww_band_halfwidth(s.net_Γ, s.S, σ, config)

    if abs(net_Δ) <= H
        return no_trade_idx()
    else
        # Trade toward zero but only to the band edge
        # If net_Δ > H, target is +H (band edge on positive side)
        # If net_Δ < -H, target is -H (band edge on negative side)
        target = sign(net_Δ) * H
        # Clamp to our grid range (±0.3)
        target = clamp(target, config.Δ_targets[2], config.Δ_targets[end])
        return nearest_hedge_idx(target, config)
    end
end

function leland_hedge_idx(
    env::EnvironmentState,
    portfolio::Portfolio,
    config::SimConfig,
    σ::Float64
)::Int
    s = env.agent_state
    σ̂ = leland_modified_vol(σ, config)

    # Compute Leland target delta for the option book
    # sum over all options: q_i × bs_Δ(S, K_i, τ, σ̂)
    opt = env.current_options[1]  # single option for now
    q = portfolio.option_quantities[1]
    Δ_leland_per_contract = bs_Δ_Γ(s.S, opt.K, s.τ, σ̂, config.r; call=opt.is_call).Δ
    Δ_options_leland = q * Δ_leland_per_contract
    Δ_options_std = q * bs_Δ_Γ(s.S, opt.K, s.τ, σ, config.r; call=opt.is_call).Δ
    target_net_Δ = Δ_options_std - Δ_options_leland

    target_net_Δ = clamp(target_net_Δ, config.Δ_targets[2], config.Δ_targets[end])
    return nearest_hedge_idx(target_net_Δ, config)
end

function naive_hedge_idx(config::SimConfig)::Int
    return nearest_hedge_idx(0.0, config)
end

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

function as_ww_policy(
    env::EnvironmentState,
    portfolio::Portfolio,
    config::SimConfig,
    σ::Float64
)::MarketMakingAction
    return MarketMakingAction(
        as_spread_idx(env, config, σ),
        ww_hedge_idx(env, config, σ)
    )
end

# spread_level_idx: which fixed spread level to always use (e.g. 2 for the $0.10 level)
function symmetric_naive_policy(
    env::EnvironmentState,
    portfolio::Portfolio,
    config::SimConfig,
    σ::Float64;
    spread_level_idx::Int = 2
)::MarketMakingAction
    return MarketMakingAction(
        symmetric_spread_idx(spread_level_idx),
        naive_hedge_idx(config)
    )
end

# ============================================================
# Section 6: Benchmark Runner
# ============================================================
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
    all_hedge_acts = Bool[]      # true = traded (not :no_trade)
    all_net_Δ      = Float64[]

    for ep in 1:n_episodes
        # Initialize fresh episode
        env       = EnvironmentState(
            AgentState(config.S0, config.T_option * config.Δt, 0.0, 0.0, 0.0, 0.0,
                       fill(1.0 / length(vol_model.σ_levels), length(vol_model.σ_levels))),
            VolState(vol_model),
            OptionContract[OptionContract(round(config.S0), true)],
            0
        )
        portfolio = Portfolio()
        push!(portfolio.option_quantities, 0)

        initialize_episode!(env, portfolio, vol_model, config; level=level)

        ep_reward = 0.0
        done = false

        while !done
            σ = σ_fn(env)
            action = policy_fn(env, portfolio, config, σ)

            # Track spread choice
            push!(all_spread_idx, Float64(action.spread_idx))

            # Track whether a hedge trade was made
            pushed_hedge = config.Δ_targets[action.hedge_idx] != :no_trade
            push!(all_hedge_acts, pushed_hedge)

            # Track current net delta before step
            push!(all_net_Δ, abs(env.agent_state.net_Δ))

            next_state, reward, done, _, _ = step_environment!(
                env, portfolio, action, config, rng; level=level
            )
            ep_reward += reward
        end

        episode_pnl[ep] = ep_reward
    end

    μ = mean(episode_pnl)
    σ_pnl = std(episode_pnl)
    sharpe = σ_pnl > 1e-10 ? μ / σ_pnl : 0.0

    return (
        episode_pnl    = episode_pnl,
        sharpe         = sharpe,
        mean_spread_idx = mean(all_spread_idx),
        hedge_freq     = mean(all_hedge_acts),
        mean_abs_net_Δ = mean(all_net_Δ)
    )
end