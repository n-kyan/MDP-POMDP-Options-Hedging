using Random
using StatsBase
include("1_types.jl")

struct VolModel
    σ_levels::Vector{Float64}
    stationary_dist::Vector{Float64}
    transition_matrix::Matrix{Float64}
    # initial_regime::Int # Should be a random var that follows a dist derived from the transition matrix not a given int value

    function VolModel(
        σ_levels::Vector{Float64};
        transition_matrix::Matrix{Float64} = ones(1, 1),
    )
        n = length(σ_levels)

        # Validations
        # all volatilities positive
        all(σ .> 0.0 for σ in σ_levels) || error("All σ_levels must be positive")

        # transition matrix is square and matches number of regimes
        size(transition_matrix) == (n, n) || error(
            "Transition matrix must be $(n)×$(n), got $(size(transition_matrix))"
        )

        # rows sum to 1 (row-stochastic)
        for i in 1:n
            row_sum = sum(transition_matrix[i, :])
            isapprox(row_sum, 1.0; atol=1e-9) || error(
                "Row $i of transition matrix sums to $row_sum, not 1.0"
            )
        end

        # all entries non-negative
        all(transition_matrix .>= 0.0) || error("Transition matrix entries must be non-negative")

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
            error("3+ regimes not yet implemented in VolModel. Use 1-2 regimes. ")
        end
        new(σ_levels, transition_matrix, π)
    end
end

mutable struct VolState
    vm::VolModel # stationary dist from VolModel
    σ::Float64# initial regime needs to come from a random sample from the dist in VolModel. Varibel hold vol of current regime

    function VolState(
        vm::VolModel
    )
        init_regime = sample(1:length(vm.σ_levels), Weights(vm.stationary_dist))

        new(vm, init_regime)
    end
end

# =================================================
# Using the above structs to model each timestep
# =================================================

# Constant Vol Version
function step_price(
    S::Float64,
    vol_state::VolState,
    config::SimConfig,
    rng::AbstractRNG
)
    σ = vol_state.σ
    Δt = config.Δt
    r = config.r

    # Draw random shock
    Z = randn(rng)

    # GBM log-return step
    S_new = S * exp((r - 0.5 * σ^2) * Δt + σ * sqrt(Δt) * Z)

    return S_new, vol_state  # vol state doesn't change for constant vol
end