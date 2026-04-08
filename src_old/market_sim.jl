# ============================================================================
# Module 2: MarketSim — Spot Price Simulation
# ============================================================================
# This module generates simulated price paths for the underlying asset.
# It provides two volatility models:
#   1. ConstantVol — standard GBM with fixed σ (Level 1)
#   2. RegimeSwitchingVol — hidden Markov chain governs σ (Level 2+)
#
# Design principle: the single-step function `step_price` is the primitive.
# The full-path function `simulate_path` is just a loop over single steps.
# The MDP environment will call `step_price`; evaluation/plotting uses `simulate_path`.
# ============================================================================

using Random
using Distributions  # for Categorical distribution (regime transitions)
using LinearAlgebra  # for eigen() in stationary_distribution (general N-regime case)

# ────────────────────────────────────────────────────────────────────────────
# Volatility Models
# ────────────────────────────────────────────────────────────────────────────

"""
    ConstantVol(σ)

Constant volatility model. The volatility never changes.
Used for Level 1 (basic GBM) experiments.

# Fields
- `σ::Float64` — annualized volatility (e.g., 0.20 for 20%)
"""
struct ConstantVol
    σ::Float64

    function ConstantVol(σ::Float64)
        σ > 0.0 || error("Volatility must be positive, got σ = $σ")
        new(σ)
    end
end

"""
    RegimeSwitchingVol(σ_levels, transition_matrix, initial_regime)

Regime-switching volatility model. A hidden Markov chain transitions between
discrete volatility regimes at each timestep.

# Fields
- `σ_levels::Vector{Float64}` — volatility for each regime (e.g., [0.15, 0.35])
- `transition_matrix::Matrix{Float64}` — row-stochastic Markov transition matrix.
    `T[i, j]` = probability of moving from regime i to regime j.
- `initial_regime::Int` — starting regime index (1-based)

# Example
A three-regime model calibrated to S&P 500 daily returns (Cerboni Baiardi et al., 2020):
```julia
vol = RegimeSwitchingVol(
    [0.10, 0.20, 0.40],     # low ≈ 10%, medium ≈ 20%, high ≈ 40% annualized vol
    [0.977 0.023 0.000;     # low:    stays low 97.7%, moves to medium 2.3%
     0.042 0.952 0.006;     # medium: moves to low 4.2%, stays medium 95.2%, moves to high 0.6%
     0.000 0.027 0.973],    # high:   moves to medium 2.7%, stays high 97.3%
    1                        # start in low regime
)
```
Note the near-zero off-diagonals between low and high: empirically, volatility
transitions are sequential — markets pass through the medium regime rather than
jumping directly between calm and crisis.
"""
struct RegimeSwitchingVol
    σ_levels::Vector{Float64}
    transition_matrix::Matrix{Float64}
    initial_regime::Int

    function RegimeSwitchingVol(
        σ_levels::Vector{Float64},
        transition_matrix::Matrix{Float64},
        initial_regime::Int
    )
        n = length(σ_levels)

        # Validate: all volatilities positive
        all(σ .> 0.0 for σ in σ_levels) || error("All σ_levels must be positive")

        # Validate: transition matrix is square and matches number of regimes
        size(transition_matrix) == (n, n) || error(
            "Transition matrix must be $(n)×$(n), got $(size(transition_matrix))"
        )

        # Validate: rows sum to 1 (row-stochastic)
        for i in 1:n
            row_sum = sum(transition_matrix[i, :])
            isapprox(row_sum, 1.0; atol=1e-10) || error(
                "Row $i of transition matrix sums to $row_sum, not 1.0"
            )
        end

        # Validate: all entries non-negative
        all(transition_matrix .>= 0.0) || error("Transition matrix entries must be non-negative")

        # Validate: initial regime is valid
        1 <= initial_regime <= n || error(
            "initial_regime must be between 1 and $n, got $initial_regime"
        )

        new(σ_levels, transition_matrix, initial_regime)
    end
end

# ────────────────────────────────────────────────────────────────────────────
# Calibrated Parameters (S&P 500 / ES Options)
# ────────────────────────────────────────────────────────────────────────────

"""
    CALIBRATED_3STATE_VOL

Pre-built RegimeSwitchingVol calibrated to S&P 500 daily returns.

Regime definitions (annualized realized volatility):
  1 = Low    (σ ≈ 10%) — calm market, VIX typically below 17
  2 = Medium (σ ≈ 20%) — elevated uncertainty, VIX roughly 17–28
  3 = High   (σ ≈ 40%) — crisis/stress, VIX above 28

Transition matrix derivation: off-diagonal entries are chosen to reproduce the
target stationary distribution π ≈ [0.60, 0.33, 0.07] (Cerboni Baiardi et al.,
2020, Risks MDPI 8(3) 71; corroborated by VIX empirical percentiles) while
preserving empirically calibrated mean regime durations (~43 days low, ~24 days
medium, ~37 days high). Given the no-skip constraint (low↔high = 0), the
stationary equations reduce to:
  π₁·a = π₂·b  →  b = (0.60/0.33)·0.023 ≈ 0.042
  π₃·d = π₂·c  →  c = (0.07/0.33)·0.027 ≈ 0.006

Key property: near-zero probability of jumping directly between low and high
regimes — transitions are sequential through the medium state.
"""
const CALIBRATED_3STATE_VOL = RegimeSwitchingVol(
    [0.10, 0.20, 0.40],        # σ_low, σ_mid, σ_high (annualized)
    [0.977  0.023  0.000;      # low    → low 97.7%, medium 2.3%, high 0.0%
     0.042  0.952  0.006;      # medium → low 4.2%,  medium 95.2%, high 0.6%
     0.000  0.027  0.973],     # high   → low 0.0%,  medium 2.7%, high 97.3%
    1                           # start in low regime (sampled from stationary in practice)
)

# ────────────────────────────────────────────────────────────────────────────
# Regime Helpers
# ────────────────────────────────────────────────────────────────────────────

"""
    stationary_distribution(vol::RegimeSwitchingVol) → Vector{Float64}

Compute the stationary distribution of the regime Markov chain.

For a 2-regime model with transition matrix [p 1-p; 1-q q], the stationary
distribution is π = [(1-q)/(2-p-q), (1-p)/(2-p-q)].

For N regimes, solves π'T = π' with Σπ = 1 by finding the left eigenvector
of T with eigenvalue 1.
"""
function stationary_distribution(vol::RegimeSwitchingVol)
    T = vol.transition_matrix
    n = size(T, 1)

    if n == 2
        # Closed-form for 2 regimes (faster and numerically exact)
        p_leave_1 = 1.0 - T[1, 1]   # P(leave regime 1)
        p_leave_2 = 1.0 - T[2, 2]   # P(leave regime 2)
        total = p_leave_1 + p_leave_2
        return [p_leave_2 / total, p_leave_1 / total]
    else
        # General case: solve via eigendecomposition of T'
        # The stationary distribution is the left eigenvector with eigenvalue 1
        # which is the right eigenvector of T'
        vals, vecs = eigen(T')
        # Find the eigenvalue closest to 1.0
        idx = argmin(abs.(vals .- 1.0))
        π = real.(vecs[:, idx])
        π ./= sum(π)   # normalize to sum to 1
        return π
    end
end

"""
    sample_initial_regime(vol::RegimeSwitchingVol, rng::AbstractRNG) → Int

Sample a starting regime from the stationary distribution.
Use this when initializing episodes so the agent doesn't always start
in the same regime.
"""
function sample_initial_regime(vol::RegimeSwitchingVol, rng::AbstractRNG)
    π = stationary_distribution(vol)
    return rand(rng, Categorical(π))
end

"""
    with_initial_regime(vol::RegimeSwitchingVol, regime::Int) → RegimeSwitchingVol

Create a copy of the vol model with a different initial regime.
This is how the environment will randomize starting regimes per episode.
"""
function with_initial_regime(vol::RegimeSwitchingVol, regime::Int)
    return RegimeSwitchingVol(vol.σ_levels, vol.transition_matrix, regime)
end

"""
    MarketParams{V}(S0, r, dt, vol)

Top-level container for all simulation parameters.
Parameterized by the volatility model type `V`, which enables Julia's
multiple dispatch to generate specialized, fast code for each model.

# Fields
- `S0::Float64`  — initial spot price
- `r::Float64`   — risk-free rate (annualized). Used as drift (risk-neutral measure).
- `dt::Float64`  — timestep size in years (1/252 for daily)
- `vol::V`       — volatility model (ConstantVol or RegimeSwitchingVol)

# Examples
```julia
# Constant vol (Level 1)
params = MarketParams(100.0, 0.05, 1/252, ConstantVol(0.20))

# Regime switching (Level 2+)
vol = RegimeSwitchingVol([0.15, 0.35], [0.98 0.02; 0.05 0.95], 1)
params = MarketParams(100.0, 0.05, 1/252, vol)
```
"""
struct MarketParams{V}
    S0::Float64
    r::Float64
    dt::Float64
    vol::V

    function MarketParams(S0::Float64, r::Float64, dt::Float64, vol::V) where V
        S0 > 0.0 || error("Initial price must be positive, got S0 = $S0")
        dt > 0.0 || error("Timestep must be positive, got dt = $dt")
        new{V}(S0, r, dt, vol)
    end
end

# ────────────────────────────────────────────────────────────────────────────
# Volatility State
# ────────────────────────────────────────────────────────────────────────────
# The "volatility state" tracks whatever internal state the vol model needs
# between timesteps. For ConstantVol, there's nothing to track. For
# RegimeSwitchingVol, we need to know the current regime.

"""
    VolState

Tracks the internal state of the volatility model between timesteps.
- For ConstantVol: no state needed (empty struct)
- For RegimeSwitchingVol: tracks the current regime index
"""
struct ConstantVolState end

struct RegimeVolState
    regime::Int # this is fine but needs to come from a random sample from the dist in RegimeSwitchingVol
end

# Initialize vol state from the vol model
init_vol_state(vol::ConstantVol) = ConstantVolState()
init_vol_state(vol::RegimeSwitchingVol) = RegimeVolState(vol.initial_regime)

# Get current σ from vol state
get_σ(vol::ConstantVol, vs::ConstantVolState) = vol.σ
get_σ(vol::RegimeSwitchingVol, vs::RegimeVolState) = vol.σ_levels[vs.regime]

# ────────────────────────────────────────────────────────────────────────────
# Single-Step Price Update (THE CORE PRIMITIVE)
# ────────────────────────────────────────────────────────────────────────────

"""
    step_price(S, vol_state, params, rng) → (S_new, vol_state_new)

Advance the price by one timestep.

This is the fundamental building block. The MDP environment's `gen()` function
will call this. The full-path simulator is just a loop over this.

# Returns
- `S_new::Float64` — price after one timestep
- `vol_state_new` — updated volatility state

# Price Update Formula (GBM, log-return form)
    S_{t+1} = S_t × exp((r - σ²/2) × dt + σ × √dt × Z)
    where Z ~ N(0, 1)

The -σ²/2 is the Itô correction that ensures E[S_{t+1}] = S_t × exp(r × dt).
"""
function step_price(
    S::Float64,
    vol_state::ConstantVolState,
    params::MarketParams{ConstantVol},
    rng::AbstractRNG
)
    σ = params.vol.σ
    dt = params.dt
    r = params.r

    # Draw random shock
    Z = randn(rng)

    # GBM log-return step
    S_new = S * exp((r - 0.5 * σ^2) * dt + σ * sqrt(dt) * Z)

    return S_new, vol_state  # vol state doesn't change for constant vol
end

function step_price(
    S::Float64,
    vol_state::RegimeVolState,
    params::MarketParams{RegimeSwitchingVol},
    rng::AbstractRNG
)
    vol = params.vol
    dt = params.dt
    r = params.r

    # Step 1: Transition to new regime (may or may not switch)
    # Sample from the row of the transition matrix corresponding to current regime
    transition_probs = vol.transition_matrix[vol_state.regime, :]
    new_regime = rand(rng, Categorical(transition_probs))

    # Step 2: Get volatility for the new regime
    σ = vol.σ_levels[new_regime]

    # Step 3: GBM step with this regime's volatility
    Z = randn(rng)
    S_new = S * exp((r - 0.5 * σ^2) * dt + σ * sqrt(dt) * Z)

    return S_new, RegimeVolState(new_regime)
end

# ────────────────────────────────────────────────────────────────────────────
# Full Path Simulation
# ────────────────────────────────────────────────────────────────────────────

"""
    SimulationResult

Container for a complete simulation run. Stores everything needed for
analysis, plotting, and debugging.

# Fields
- `prices::Vector{Float64}`   — price at each step (length n_steps + 1, includes S0)
- `regimes::Vector{Int}`      — regime at each step (length n_steps + 1).
                                 All 1s for ConstantVol.
- `log_returns::Vector{Float64}` — log(S_{t+1}/S_t) at each step (length n_steps)
- `params::MarketParams`      — the parameters used to generate this path
"""
struct SimulationResult{V}
    prices::Vector{Float64}
    regimes::Vector{Int}
    log_returns::Vector{Float64}
    params::MarketParams{V}
end

"""
    simulate_path(params, n_steps, rng; randomize_start=true) → SimulationResult

Generate a complete price path by looping over `step_price`.

# Arguments
- `params::MarketParams` — simulation parameters
- `n_steps::Int`         — number of timesteps to simulate
- `rng::AbstractRNG`     — random number generator (for reproducibility)
- `randomize_start::Bool` — if true (default), sample the initial regime from the
                            stationary distribution rather than using `vol.initial_regime`.
                            Set to false only when you explicitly need a fixed starting regime
                            (e.g., in unit tests or when running from a known state).

# Why randomize_start=true matters
If the agent always starts in the same regime, it will learn a policy that is
implicitly conditioned on the starting regime — exploiting an artifact of the
simulation rather than learning a truly general policy. Sampling the initial
regime from the stationary distribution ensures each episode is i.i.d. from
the perspective of the long-run regime distribution.
"""
function simulate_path(
    params::MarketParams,
    n_steps::Int,
    rng::AbstractRNG;
    randomize_start::Bool = true
)
    # Pre-allocate arrays
    prices = Vector{Float64}(undef, n_steps + 1)
    regimes = Vector{Int}(undef, n_steps + 1)
    log_returns = Vector{Float64}(undef, n_steps)

    # Initialize vol state — randomize starting regime if requested
    vol_state = if randomize_start && params.vol isa RegimeSwitchingVol
        start_regime = sample_initial_regime(params.vol, rng)
        RegimeVolState(start_regime)
    else
        init_vol_state(params.vol)
    end

    # Initialize price
    prices[1] = params.S0
    regimes[1] = _get_regime(vol_state)

    # Simulate step by step
    for t in 1:n_steps
        S_new, vol_state = step_price(prices[t], vol_state, params, rng)
        prices[t + 1] = S_new
        regimes[t + 1] = _get_regime(vol_state)
        log_returns[t] = log(prices[t + 1] / prices[t])
    end

    return SimulationResult(prices, regimes, log_returns, params)
end

# Helper to extract regime index from vol state
_get_regime(::ConstantVolState) = 1
_get_regime(vs::RegimeVolState) = vs.regime

# ────────────────────────────────────────────────────────────────────────────
# Convenience: simulate with a seed (for quick testing)
# ────────────────────────────────────────────────────────────────────────────

"""
    simulate_path(params, n_steps; seed=42, randomize_start=true) → SimulationResult

Convenience method that creates an RNG from a seed.
Use the explicit `rng` version for production/evaluation code.
"""
function simulate_path(params::MarketParams, n_steps::Int; seed::Int=42, randomize_start::Bool=true)
    rng = MersenneTwister(seed)
    return simulate_path(params, n_steps, rng; randomize_start=randomize_start)
end