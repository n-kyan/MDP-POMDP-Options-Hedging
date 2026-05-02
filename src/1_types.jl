using Random
using StatsBase: sample, Weights

Base.@kwdef struct SimConfig
    # --- Market parameters ---
    S0::Float64 = 100.0
    r::Float64 = 0.05          # risk-free rate (annualized)
    Δt::Float64 = 1 / 252        # timestep in years (1 trading day)

    # --- Option parameters ---
    T_option::Int = 30         # trading days per option lifetime (~1 month)
    n_options_per_episode::Int = 5

    # --- Transaction costs ---
    κ::Float64 = 0.001         # proportional hedge cost (10 bps)

    # --- Fill model (AS 2008) ---
    A::Float64 = 140.0         # fill intensity scale
    k::Float64 = 6.0           # fill decay rate
    γ_market::Float64 = 0.1   # risk aversion used in GLF-T benchmark formula

    # --- Reward ---
    φ::Float64 = 0.01          # agent risk aversion (delta penalty weight)
end

# ---- Volatility model -------------------------

struct VolModel
    σ_levels::Vector{Float64}
    transition_matrix::Matrix{Float64}
    stationary_dist::Vector{Float64}

    function VolModel(
        σ_levels::Vector{Float64};
        transition_matrix::Matrix{Float64}=ones(1, 1),
    )
        n = length(σ_levels)
        all(σ .> 0.0 for σ in σ_levels) || error("All σ_levels must be positive")
        size(transition_matrix) == (n, n) || error("Transition matrix must be $(n)×$(n)")
        all(transition_matrix .>= 0.0) || error("Transition matrix entries must be non-negative")
        for i in 1:n
            isapprox(sum(transition_matrix[i, :]), 1.0; atol=1e-9) ||
                error("Row $i of transition matrix does not sum to 1")
        end

        if n == 1
            π = [1.0]
        elseif n == 2
            p12 = 1.0 - transition_matrix[1, 1]
            p21 = 1.0 - transition_matrix[2, 2]
            denom = p12 + p21
            π = denom ≈ 0.0 ? [0.5, 0.5] : [p21 / denom, p12 / denom]
        else
            error("3+ regimes not implemented. Use 1-2 regimes.")
        end
        new(σ_levels, transition_matrix, π)
    end
end

struct VolState
    vm::VolModel
    regime_idx::Int

    function VolState(vm::VolModel)
        regime_idx = sample(1:length(vm.σ_levels), Weights(vm.stationary_dist))
        new(vm, regime_idx)
    end

    function VolState(vm::VolModel, regime_idx::Int)
        new(vm, regime_idx)
    end
end

get_σ(vs::VolState) = vs.vm.σ_levels[vs.regime_idx]

# Returns the transition-row belief which is what the market uses to price options.
function perfect_regime_belief(vs::VolState)
    return vs.vm.transition_matrix[vs.regime_idx, :]
end

# ---- Option and portfolio structs --------------------------------

struct OptionContract
    K::Float64
    is_call::Bool
end

mutable struct Portfolio
    option_quantities::Vector{Int}
    q_spot::Float64
    cash::Float64

    Portfolio() = new(Int[], 0.0, 0.0)
end

# ---- Actions -----------------------------------------------------

# Continuous action: half-spread δ and target portfolio delta Δ_target.
#
# Economic bounds (enforced by policy, not the environment):
#   δ       ∈ [κ·S·|hat_Δ̂,  hat_V]     — covers hedge cost; below option value
#   Δ_target ∈ [min(0, hat_Δ_P), max(0, hat_Δ_P)]  — reduce |Δ| without flipping sign
struct MarketMakingAction
    δ::Float64          # half-spread in dollars
    Δ_target::Float64   # desired net portfolio delta after hedging
end

# ---- Agent state (observation + belief-derived quantities) -------

# o_t = (r_t, hat_Δ_P, hat_Γ_P, f_t, τ) per the paper.
# S, σ_hat, hat_V, hat_Δ are included because policies need them to compute action bounds.
struct AgentState
    # Core observation
    r_t::Float64        # log return this step (NaN at episode start)
    hat_Δ_P::Float64    # net portfolio delta under σ_hat (options + spot position)
    hat_Γ_P::Float64    # portfolio gamma under σ_hat
    f_t::Int            # fill indicator: +1 bid only, -1 ask only, 0 no/both
    τ::Float64          # time to expiry (years)

    # Belief and action-bound quantities
    σ_hat::Float64      # vol estimate
    hat_V::Float64      # believed option value under σ_hat
    hat_Δ::Float64      # per-contract delta under σ_hat (for lower bound of δ)
    S::Float64          # spot price (for action lower bound κ·S·|hat_Δ|)
end

# ---- Environment state -------------------------------------------

mutable struct EnvironmentState
    agent_state::AgentState
    vol_state::VolState
    current_options::Vector{OptionContract}
    options_completed::Int
end

# ---- Particle filter (belief over σ) -----------------------------

# Maintains a particle approximation of p(σ* | history).
# Stored in log-space for numerical stability.
# Weights are stored as log-unnormalized; call get_weights() to normalize.
mutable struct ParticleFilter
    log_σ::Vector{Float32}   # log of each particle's σ estimate
    log_w::Vector{Float64}   # log unnormalized weights (avoids underflow)
    n::Int
end

# ---- Step information --------------------------------------------

struct FillOutcome
    bid_filled::Bool
    ask_filled::Bool
    bid_price::Float64
    ask_price::Float64
    V_market::Float64
    f_t::Int   # +1 bid only, -1 ask only, 0 no fill or simultaneous fills
end

struct StepInfo
    log_return::Float64
    fill::FillOutcome
    shares_traded::Float64
    hedge_cost::Float64
    wealth_before::Float64
    wealth_after::Float64
end

# ---- POMDP observation type (shared by Module 9 and 10) ----------

# Minimal observation for vol inference: log-return (continuous vol signal)
# and fill indicator (discrete directional signal).
# S, τ, and portfolio quantities are fully observable from state and excluded.
struct OptionsMMObs
    r_t::Float64
    f_t::Int     # +1 bid only, -1 ask only, 0 otherwise
end
