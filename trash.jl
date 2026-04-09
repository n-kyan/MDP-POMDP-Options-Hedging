using Random
using StatsBase: sample, Weights

# ────────────────────────────────────────────────────────────────────────────
# Volatility Model + State (unchanged from current)
# ────────────────────────────────────────────────────────────────────────────

struct VolModel
    σ_levels::Vector{Float64}
    transition_matrix::Matrix{Float64}
    stationary_dist::Vector{Float64}

    function VolModel(
        σ_levels::Vector{Float64};
        transition_matrix::Matrix{Float64} = ones(1, 1),
    )
        n = length(σ_levels)

        # Validations
        all(σ .> 0.0 for σ in σ_levels) || error(
            "All σ_levels must be positive, got $σ_levels"
        )
        size(transition_matrix) == (n, n) || error(
            "Transition matrix must be $(n)×$(n), got $(size(transition_matrix))"
        )
        all(transition_matrix .>= 0.0) || error(
            "Transition matrix entries must be non-negative"
        )
        for i in 1:n
            row_sum = sum(transition_matrix[i, :])
            isapprox(row_sum, 1.0; atol=1e-9) || error(
               "Row $i of transition matrix sums to $(sum(transition_matrix[i, :])), not 1.0"
            )
        end

        if n == 1
            π = [1.0]
        elseif n == 2
            p1_1 = transition_matrix[1, 1]
            p2_2 = transition_matrix[2, 2]
            p1_2 = 1.0 - p1_1
            p2_1 = 1.0 - p2_2
            denominator = p1_2 + p2_1
            if denominator ≈ 0.0
                π = [0.5, 0.5]
            else
                π = [p2_1/denominator, p1_2/denominator]
            end
        else
            error("3+ regimes not yet implemented in VolModel for this project. Use 1-2 regimes.")
        end
        new(σ_levels, transition_matrix, π,)
    end
end

mutable struct VolState
    vm::VolModel
    regime_idx::Int

    function VolState(vm::VolModel)
        init_regime = sample(1:length(vm.σ_levels), Weights(vm.stationary_dist))
        new(vm, init_regime)
    end
end

get_σ(vm::VolModel, vs::VolState) = vm.σ_levels[vs.regime_idx]


# ────────────────────────────────────────────────────────────────────────────
# Option Contract
# ────────────────────────────────────────────────────────────────────────────

struct OptionContract
    K::Float64
    is_call::Bool
end


# ────────────────────────────────────────────────────────────────────────────
# Simulation Configuration
# ────────────────────────────────────────────────────────────────────────────

Base.@kwdef struct SimConfig
    # --- Market parameters ---
    S0::Float64 = 100.0              # initial spot price
    r::Float64 = 0.05                # risk-free rate (annualized)
    Δt::Float64 = 1/252              # timestep in years (1 trading day)

    # --- Option parameters ---
    T_option::Int = 63               # trading days per option lifetime
    n_options_per_episode::Int = 8    # sequential options per training episode

    # --- Transaction costs ---
    κ::Float64 = 0.001               # proportional cost (10 bps)

    # --- Fill model (AS 2008) ---
    A::Float64 = 140.0               # fill intensity  
    k::Float64 = 6.0                 # fill decay rate (calibrated for options)

    # --- Reward ---
    φ::Float64 = 0.01                # risk aversion (inventory penalty weight)

    # --- Action space ---
    spread_levels::Vector{Float64} = [0.05, 0.10, 0.20, 0.40, 0.80]
    hedge_targets::Vector{Float64} = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25]
end

# Total number of discrete actions (spread levels × hedge targets).
n_actions(config::SimConfig) = length(config.spread_levels) * length(config.hedge_targets)


# ────────────────────────────────────────────────────────────────────────────
# Market-Making Action
# ────────────────────────────────────────────────────────────────────────────

struct MarketMakingAction
    spread_idx::Int
    hedge_idx::Int
end

function action_from_index(i::Int, config::SimConfig)
    n_hedge = length(config.hedge_targets)
    spread_idx = div(i - 1, n_hedge) + 1
    hedge_idx = mod(i - 1, n_hedge) + 1
    return MarketMakingAction(spread_idx, hedge_idx)
end

function action_to_index(a::MarketMakingAction, config::SimConfig)
    n_hedge = length(config.hedge_targets)
    return (a.spread_idx - 1) * n_hedge + a.hedge_idx
end


# ────────────────────────────────────────────────────────────────────────────
# State Representations
# ────────────────────────────────────────────────────────────────────────────

#=
HedgingState — what the agent observes.

The `regime_belief` field has different meanings at each level:
  - Level 1: single element [1.0] (constant vol, no uncertainty)
  - Level 2: one-hot at true regime, e.g. [1.0, 0.0] (agent knows regime)
  - Level 3: probability distribution over regimes, updated by Hamilton 
    filter using returns + fill asymmetry. This is the agent's imperfect 
    estimate of where the market is pricing.
=#
struct HedgingState
    S::Float64
    τ::Float64
    q_calls::Int
    q_puts::Int
    q_spot::Int
    cash::Float64
    regime_belief::Vector{Float64}
end

#=
EnvironmentState — the full simulation state including hidden information.

The market has perfect knowledge of the current regime. V_market is computed
by environment.jl using the true regime and transition probabilities:

    V_market = Σⱼ P(regime_next = j | regime_current = i) × V_BS(σⱼ)

This is the transition-row-weighted BS price, which is the correct 
one-step-ahead expected value for a market participant with perfect 
regime knowledge. The market and "God" are the same entity in our model.

The agent does NOT see vol_state directly. In Level 3, the agent must 
infer V_market from the fill asymmetry pattern — fills are computed 
against V_market, so when the agent's V_believed diverges from V_market,
fills become asymmetric, nudging the agent's belief toward the market's
pricing.

No market_belief field is needed since the market has perfect knowledge.
No inventory limit Q — inventory risk is managed via the reward penalty 
φ·Δ_net², following our decision to remove hard AS-style inventory bounds.
=#
mutable struct EnvironmentState
    agent_state::HedgingState
    vol_state::VolState
    current_options::Vector{OptionContract}
    options_completed::Int
end


# ────────────────────────────────────────────────────────────────────────────
# V_market computation helper
# ────────────────────────────────────────────────────────────────────────────

#=
compute_market_belief(vs::VolState) → Vector{Float64}

Returns the probability distribution over regimes that the market uses 
for pricing. Since the market has perfect regime knowledge, this is 
simply the current regime's row of the transition matrix — reflecting 
that the market knows the current regime and accounts for the possibility 
of transitioning next step.

This is used by environment.jl to compute V_market via bs_all_belief_weighted.
=#
function compute_market_belief(vs::VolState)
    return vs.vm.transition_matrix[vs.regime_idx, :]
end