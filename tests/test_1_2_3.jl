# ============================================================================
# test_modules_1_2_3.jl — Smoke tests for types, BS, and spot dynamics
# ============================================================================
# Run with: julia test_modules_1_2_3.jl
# ============================================================================

include("../src/1_types.jl")
include("../src/2_black_scholes.jl")
include("../src/3_spot_dynamics.jl")

using Printf
using Statistics

passed = 0
failed = 0

function check(name::String, condition::Bool)
    global passed, failed
    if condition
        passed += 1
        println("  ✓ $name")
    else
        failed += 1
        println("  ✗ FAIL: $name")
    end
end

# ════════════════════════════════════════════════════════════════════════════
# TEST 1: VolModel — Constant Vol
# ════════════════════════════════════════════════════════════════════════════
println("\n═══ TEST 1: VolModel — Constant Vol ═══")

vm_const = VolModel([0.20])
check("1 regime", length(vm_const.σ_levels) == 1)
check("stationary dist = [1.0]", vm_const.stationary_dist ≈ [1.0])
check("σ_levels correct", vm_const.σ_levels == [0.20])

# ════════════════════════════════════════════════════════════════════════════
# TEST 2: VolModel — 2-Regime (Hardy 2001)
# ════════════════════════════════════════════════════════════════════════════
println("\n═══ TEST 2: VolModel — 2-Regime Hardy ═══")

vm_hardy = VolModel(
    [0.121, 0.269],
    transition_matrix = [0.9982 0.0018;
                         0.0022 0.9978]
)
check("2 regimes", length(vm_hardy.σ_levels) == 2)

# Stationary distribution: π₁ = p21/(p12+p21), π₂ = p12/(p12+p21)
# p12 = 0.0018, p21 = 0.0022
expected_π1 = 0.0022 / (0.0018 + 0.0022)  # ≈ 0.55
expected_π2 = 0.0018 / (0.0018 + 0.0022)  # ≈ 0.45
check("π₁ ≈ 0.55", isapprox(vm_hardy.stationary_dist[1], expected_π1, atol=0.01))
check("π₂ ≈ 0.45", isapprox(vm_hardy.stationary_dist[2], expected_π2, atol=0.01))
check("π sums to 1", isapprox(sum(vm_hardy.stationary_dist), 1.0, atol=1e-10))

@printf("  Stationary dist: [%.4f, %.4f]\n", vm_hardy.stationary_dist[1], vm_hardy.stationary_dist[2])

# ════════════════════════════════════════════════════════════════════════════
# TEST 3: VolModel — Validation
# ════════════════════════════════════════════════════════════════════════════
println("\n═══ TEST 3: VolModel — Validation ═══")

try
    VolModel([-0.1])
    check("Negative σ rejected", false)
catch e
    check("Negative σ rejected", true)
end

try
    VolModel([0.15, 0.35], transition_matrix = [0.5 0.3; 0.5 0.5])
    check("Non-stochastic matrix rejected", false)
catch e
    check("Non-stochastic matrix rejected", true)
end

try
    VolModel([0.15, 0.35], transition_matrix = [0.9 0.1 0.0; 0.1 0.9 0.0])
    check("Wrong-size matrix rejected", false)
catch e
    check("Wrong-size matrix rejected", true)
end

# ════════════════════════════════════════════════════════════════════════════
# TEST 4: VolState — Initial Regime Sampling
# ════════════════════════════════════════════════════════════════════════════
println("\n═══ TEST 4: VolState — Initial Regime Sampling ═══")

rng = MersenneTwister(42)
n_samples = 10_000
regime_counts = zeros(Int, 2)
for _ in 1:n_samples
    vs = VolState(vm_hardy)
    regime_counts[vs.regime_idx] += 1
end

empirical_frac1 = regime_counts[1] / n_samples
@printf("  Empirical regime 1 fraction: %.4f (expected ≈ %.4f)\n", empirical_frac1, expected_π1)
check("Regime 1 sampled ≈ π₁ (within 3%)", abs(empirical_frac1 - expected_π1) < 0.03)

# Constant vol always starts in regime 1
vs_const = VolState(vm_const)
check("Constant vol starts in regime 1", vs_const.regime_idx == 1)
check("get_σ returns 0.20", get_σ(vm_const, vs_const) == 0.20)

# ════════════════════════════════════════════════════════════════════════════
# TEST 5: SimConfig — Construction
# ════════════════════════════════════════════════════════════════════════════
println("\n═══ TEST 5: SimConfig ═══")

config = SimConfig()
check("n_actions = 30", n_actions(config) == 30)
check("S0 default = 100", config.S0 == 100.0)
check("Δt default = 1/252", config.Δt ≈ 1/252)

# Action index round-trip
for i in 1:30
    a = action_from_index(i, config)
    check_i = action_to_index(a, config)
    if check_i != i
        check("Action index round-trip i=$i", false)
        break
    end
end
check("Action index round-trip (all 30)", true)

# ════════════════════════════════════════════════════════════════════════════
# TEST 6: Black-Scholes — Known Values
# ════════════════════════════════════════════════════════════════════════════
println("\n═══ TEST 6: Black-Scholes — Known Values ═══")

# ATM call: S=K=100, σ=0.20, τ=0.25, r=0.05
# Expected: price ≈ 3.99, delta ≈ 0.5364
S, K, τ, σ, r = 100.0, 100.0, 0.25, 0.20, 0.05

result = bs_all(S, K, τ, σ, r; call=true)
@printf("  Price: %.4f (expected ≈ 3.99)\n", result.price)
@printf("  Delta: %.4f (expected ≈ 0.5364)\n", result.Δ)
@printf("  Gamma: %.6f\n", result.Γ)
@printf("  Vega:  %.4f\n", result.ν)

check("ATM call price ≈ 3.99", isapprox(result.price, 3.99, atol=0.05))
check("ATM call delta ≈ 0.5364", isapprox(result.Δ, 0.5364, atol=0.01))
check("Gamma > 0", result.Γ > 0.0)
check("Vega > 0", result.ν > 0.0)

# Put-call parity: C - P = S - K*exp(-rτ)
put_result = bs_all(S, K, τ, σ, r; call=false)
parity_lhs = result.price - put_result.price
parity_rhs = S - K * exp(-r * τ)
@printf("  Put-call parity: C-P = %.4f, S-Ke^{-rτ} = %.4f\n", parity_lhs, parity_rhs)
check("Put-call parity holds", isapprox(parity_lhs, parity_rhs, atol=1e-10))

# Delta: call_Δ - put_Δ = 1
check("Call Δ - Put Δ ≈ 1", isapprox(result.Δ - put_result.Δ, 1.0, atol=1e-10))

# Gamma same for call and put
check("Call Γ = Put Γ", isapprox(result.Γ, put_result.Γ, atol=1e-10))

# ════════════════════════════════════════════════════════════════════════════
# TEST 7: Black-Scholes — Expiry Guards
# ════════════════════════════════════════════════════════════════════════════
println("\n═══ TEST 7: Black-Scholes — Expiry Guards ═══")

# ITM call at expiry
itm = bs_all(105.0, 100.0, 0.0, 0.20, 0.05; call=true)
check("ITM call price = intrinsic", itm.price == 5.0)
check("ITM call Δ = 1.0", itm.Δ == 1.0)
check("Expiry Γ = 0", itm.Γ == 0.0)
check("Expiry ν = 0", itm.ν == 0.0)

# OTM call at expiry
otm = bs_all(95.0, 100.0, 0.0, 0.20, 0.05; call=true)
check("OTM call price = 0", otm.price == 0.0)
check("OTM call Δ = 0.0", otm.Δ == 0.0)

# ITM put at expiry
itm_put = bs_all(95.0, 100.0, 0.0, 0.20, 0.05; call=false)
check("ITM put price = 5.0", itm_put.price == 5.0)
check("ITM put Δ = -1.0", itm_put.Δ == -1.0)

# ════════════════════════════════════════════════════════════════════════════
# TEST 8: Belief-Weighted Pricing
# ════════════════════════════════════════════════════════════════════════════
println("\n═══ TEST 8: Belief-Weighted Pricing ═══")

σ_regimes = [0.121, 0.269]

# Certain in regime 1: should match bs_all with σ=0.121
certain_low = bs_all_belief_weighted(100.0, 100.0, 0.25, σ_regimes, [1.0, 0.0], 0.05)
direct_low = bs_all(100.0, 100.0, 0.25, 0.121, 0.05)
check("Certain low-vol matches direct", isapprox(certain_low.price, direct_low.price, atol=1e-10))

# Certain in regime 2: should match bs_all with σ=0.269
certain_high = bs_all_belief_weighted(100.0, 100.0, 0.25, σ_regimes, [0.0, 1.0], 0.05)
direct_high = bs_all(100.0, 100.0, 0.25, 0.269, 0.05)
check("Certain high-vol matches direct", isapprox(certain_high.price, direct_high.price, atol=1e-10))

# Mixed belief: price should be between the two regimes
mixed = bs_all_belief_weighted(100.0, 100.0, 0.25, σ_regimes, [0.5, 0.5], 0.05)
check("Mixed price between regimes", direct_low.price < mixed.price < direct_high.price)

# Jensen's inequality: belief-weighted price > price at mean σ
σ_mean = 0.5 * 0.121 + 0.5 * 0.269
price_at_mean_σ = bs_price(100.0, 100.0, 0.25, σ_mean, 0.05)
@printf("  Belief-weighted price: %.4f\n", mixed.price)
@printf("  Price at mean σ:       %.4f\n", price_at_mean_σ)
check("Jensen's inequality: E[V(σ)] ≥ V(E[σ])", mixed.price >= price_at_mean_σ - 1e-10)

# ════════════════════════════════════════════════════════════════════════════
# TEST 9: Spot Dynamics — Constant Vol
# ════════════════════════════════════════════════════════════════════════════
println("\n═══ TEST 9: Spot Dynamics — Constant Vol ═══")

config_const = SimConfig()  # default constant 20% vol
rng = MersenneTwister(123)

# Collect many returns and check statistics
n_trials = 100_000
returns = Float64[]
S_curr = 100.0
vs = VolState(vm_const)
for _ in 1:n_trials
    S_new, log_ret = step(S_curr, vs, config_const, rng)
    push!(returns, log_ret)
    S_curr = S_new
end

expected_mean = (0.05 - 0.5 * 0.20^2) * (1/252)
expected_std = 0.20 * sqrt(1/252)
actual_mean = mean(returns)
actual_std = std(returns)

se = expected_std / sqrt(n_trials)
@printf("  Expected mean: %.8f, Actual: %.8f (SE: %.8f)\n", expected_mean, actual_mean, se)
@printf("  Expected std:  %.6f, Actual: %.6f\n", expected_std, actual_std)

check("Mean log-return correct (within 3 SE)", abs(actual_mean - expected_mean) < 3 * se)
check("Std log-return correct (within 1%)", abs(actual_std - expected_std) / expected_std < 0.01)
check("Regime stays at 1 (constant vol)", vs.regime_idx == 1)

# ════════════════════════════════════════════════════════════════════════════
# TEST 10: Spot Dynamics — Regime Switching
# ════════════════════════════════════════════════════════════════════════════
println("\n═══ TEST 10: Spot Dynamics — Regime Switching ═══")

config_rs = SimConfig()
rng = MersenneTwister(456)

# Run a long simulation and check regime frequencies
n_steps = 500_000
regime_time = zeros(Int, 2)
vs_rs = VolState(vm_hardy)
S_curr = 100.0
for _ in 1:n_steps
    S_curr, _ = step(S_curr, vs_rs, config_rs, rng)
    regime_time[vs_rs.regime_idx] += 1
end

frac1 = regime_time[1] / n_steps
@printf("  Regime 1 time fraction: %.4f (expected ≈ %.4f)\n", frac1, expected_π1)
check("Regime 1 time ≈ π₁ (within 2%)", abs(frac1 - expected_π1) < 0.02)
check("Both regimes visited", all(regime_time .> 0))
check("Price stayed positive", S_curr > 0.0)

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
println("\n" * "═"^50)
println("Results: $passed passed, $failed failed")
println("═"^50)