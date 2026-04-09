using Random
using StatsBase: sample, Weights

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
        # all volatilities positive
        all(σ .> 0.0 for σ in σ_levels) || error(
            "All σ_levels must be positive, got $σ_levels"
        )

        # transition matrix is square and matches number of regimes
        size(transition_matrix) == (n, n) || error(
            "Transition matrix must be $(n)×$(n), got $(size(transition_matrix))"
        )
        # all entries non-negative
        all(transition_matrix .>= 0.0) || error(
            "Transition matrix entries must be non-negative"
        )
        # rows sum to 1 (row-stochastic)
        for i in 1:n
            row_sum = sum(transition_matrix[i, :])
            isapprox(row_sum, 1.0; atol=1e-9) || error(
               "Row $i of transition matrix sums to $(sum(transition_matrix[i, :])), not 1.0"
            )
        end

        # Calculate Stationary Dist
        if n == 1
            π = [1.0] # π means stationary_dist
        elseif n == 2
            p1_1 = transition_matrix[1, 1] # P(stay in current regime)
            p2_2 = transition_matrix[2, 2] # P(stay in current regime)

            # Regime switch probabilities
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

struct VolState
    vm::VolModel # stationary dist from VolModel
    regime_idx::Int # initial regime needs to come from a random sample from the dist in VolModel.

    # Default constructor: sample from stationary distribution
    function VolState(vm::VolModel)
        regime_idx = sample(1:length(vm.σ_levels), Weights(vm.stationary_dist))
        new(vm, regime_idx)
    end

    # Explicit constructor: used when transitioning to a known regime
    function VolState(vm::VolModel, regime_idx::Int)
        new(vm, regime_idx)
    end
end

# get helper to make code more readable
get_σ(vs::VolState) = vs.vm.σ_levels[vs.regime_idx]

struct OptionContract
    K::Float64
    is_call::Bool
end

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
    φ::Float64 = 0.01                # risk aversion (delta penalty weight)

    # --- Action space ---
    spread_levels::Vector{Float64} = [0.05, 0.10, 0.20, 0.40, 0.80]
    hedge_targets::Vector{Float64} = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25]
end

# Total number of discrete actions (spread levels × hedge targets).
n_actions(config::SimConfig) = length(config.spread_levels) * length(config.hedge_targets)

# Agent's chosen action: which spread level and which hedge target.
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

struct HedgingState # Agents observable states
    S::Float64
    τ::Float64
    net_Δ::Float64 # portfolio Δ
    net_Γ::Float64 # portfolio Γ
    cash::Float64
    regime_belief::Vector{Float64} # Vector of beliefs of what is current regime
end

mutable struct EnvironmentState
    agent_state::HedgingState            # the agent's observable state
    vol_state::VolState                  # true regime (hidden in Level 3)
    current_options::Vector{OptionContract}       # the options currently being traded. Will start with calls only and add puts of the same expiry and strike later
    options_completed::Int               # count of expired options this episode
end

# Returns beliefs if you had perfect knowledge. Used by the market to calc true options value
function market_regime_belief(vs::VolState)
    return vs.vm.transition_matrix[vs.regime_idx, :]
end
