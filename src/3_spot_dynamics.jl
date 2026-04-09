include("1_types.jl")

function step_spot(
    S::Float64,
    vs::VolState,
    config::SimConfig,
    rng::AbstractRNG
)
    vm = vs.vm # get VolModel
    Δt = config.Δt
    r = config.r
    n_regimes = length(vm.σ_levels)

    # Step 1: Transition to new regime (may or may not switch)
    if n_regimes == 1
        # No regime transition needed
    else
        # Sample from the row of the transition matrix corresponding to current regime
        transition_probs = vm.transition_matrix[vs.regime_idx, :]
        vs.regime_idx = sample(rng, 1:n_regimes, Weights(transition_probs))
    end

    # Step 2: Get current vol
    σ = get_σ(vm, vs)

    # Step 3: GBM step with this regime's volatility
    Z = randn(rng)
    S_new = S * exp((r - 0.5 * σ^2) * Δt + σ * sqrt(Δt) * Z)

    # 4. Log return (used by Hamilton filter)
    log_return = log(S_new / S)

    return S_new, log_return  # vol state doesn't change for constant vol
end