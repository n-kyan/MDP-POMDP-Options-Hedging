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

    # Step 1: Get current vol
    σ = get_σ(vs)

    # Step 2: GBM step with this regime's volatility
    Z = randn(rng)
    S_new = S * exp((r - 0.5 * σ^2) * Δt + σ * sqrt(Δt) * Z)

    # Step 3: Transition to new regime (may or may not switch)
    if n_regimes == 1
        # No regime transition needed
    else
        # Sample from the row of the transition matrix corresponding to current regime
        transition_probs = vm.transition_matrix[vs.regime_idx, :]
        new_regime = sample(rng, 1:n_regimes, Weights(transition_probs))
        vs = VolState(vm, new_regime)
    end


    # Step 4: Log return (observed by agent; used by particle filter for belief update)
    log_return = log(S_new / S)

    return S_new, vs, log_return  # vol state doesn't change for constant vol
end