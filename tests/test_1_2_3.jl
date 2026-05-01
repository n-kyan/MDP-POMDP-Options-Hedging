# Tests for Modules 1 (types), 2 (Black-Scholes), 3 (spot dynamics)
# Run: julia tests/test_1_2_3.jl

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

# ════════════════════════════════════════════════════════════════
# TEST 1: SimConfig defaults
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 1: SimConfig ═══")
config = SimConfig()
check("T_option = 30",            config.T_option == 30)
check("n_options_per_episode = 5", config.n_options_per_episode == 5)
check("A = 140",                   config.A == 140.0)
check("k = 6",                     config.k == 6.0)
check("κ = 0.001",                 config.κ == 0.001)
check("φ = 0.01",                  config.φ == 0.01)

# ════════════════════════════════════════════════════════════════
# TEST 2: VolModel — constant vol
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 2: VolModel — constant vol ═══")
vm_const = VolModel([0.20])
check("1 regime",                  length(vm_const.σ_levels) == 1)
check("stationary dist = [1.0]",   vm_const.stationary_dist ≈ [1.0])
check("σ_levels correct",          vm_const.σ_levels == [0.20])

# ════════════════════════════════════════════════════════════════
# TEST 3: VolModel — Hardy (2001) two-regime calibration
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 3: VolModel — Hardy 2-regime ═══")
vm_hardy = VolModel(
    [0.121, 0.269],
    transition_matrix = [0.9982 0.0018;
                         0.0022 0.9978]
)
check("2 regimes", length(vm_hardy.σ_levels) == 2)

expected_π1 = 0.0022 / (0.0018 + 0.0022)
expected_π2 = 0.0018 / (0.0018 + 0.0022)
check("π₁ ≈ 0.55",  isapprox(vm_hardy.stationary_dist[1], expected_π1, atol = 0.01))
check("π₂ ≈ 0.45",  isapprox(vm_hardy.stationary_dist[2], expected_π2, atol = 0.01))
check("π sums to 1", isapprox(sum(vm_hardy.stationary_dist), 1.0, atol = 1e-10))

# ════════════════════════════════════════════════════════════════
# TEST 4: VolModel validation
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 4: VolModel validation ═══")
try; VolModel([-0.1]);                                     check("negative σ rejected", false)
catch; check("negative σ rejected", true) end

try; VolModel([0.15, 0.35], transition_matrix = [0.5 0.3; 0.5 0.5]);
    check("non-stochastic matrix rejected", false)
catch; check("non-stochastic matrix rejected", true) end

# ════════════════════════════════════════════════════════════════
# TEST 5: VolState sampling
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 5: VolState sampling ═══")
n_samples = 10_000
regime_counts = zeros(Int, 2)
for _ in 1:n_samples
    vs = VolState(vm_hardy)
    regime_counts[vs.regime_idx] += 1
end
empirical_frac1 = regime_counts[1] / n_samples
@printf("  Empirical regime 1 fraction: %.4f (expected ≈ %.4f)\n", empirical_frac1, expected_π1)
check("Regime 1 sampled ≈ π₁ (within 3%)", abs(empirical_frac1 - expected_π1) < 0.03)

vs_const = VolState(vm_const)
check("Constant vol starts in regime 1", vs_const.regime_idx == 1)
check("get_σ returns 0.20",             get_σ(vs_const) == 0.20)

# ════════════════════════════════════════════════════════════════
# TEST 6: MarketMakingAction (continuous)
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 6: MarketMakingAction ═══")
a = MarketMakingAction(0.15, 0.05)
check("δ field",        a.δ == 0.15)
check("Δ_target field", a.Δ_target == 0.05)

# ════════════════════════════════════════════════════════════════
# TEST 7: Black-Scholes — known values
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 7: Black-Scholes — known values ═══")
S, K, τ, σ, r = 100.0, 100.0, 0.25, 0.20, 0.05
result = bs_all(S, K, τ, σ, r; call = true)
@printf("  Price: %.4f  Delta: %.4f  Gamma: %.6f\n", result.price, result.Δ, result.Γ)

check("ATM call price ≈ 4.615",  isapprox(result.price, 4.615, atol = 0.05))
check("ATM call delta ≈ 0.5695", isapprox(result.Δ,     0.5695, atol = 0.01))
check("Gamma > 0",               result.Γ > 0.0)
check("Vega > 0",                result.ν > 0.0)

put_result  = bs_all(S, K, τ, σ, r; call = false)
parity_lhs  = result.price - put_result.price
parity_rhs  = S - K * exp(-r * τ)
check("Put-call parity",          isapprox(parity_lhs, parity_rhs, atol = 1e-10))
check("Call Δ − Put Δ ≈ 1",      isapprox(result.Δ - put_result.Δ, 1.0, atol = 1e-10))
check("Call Γ = Put Γ",           isapprox(result.Γ, put_result.Γ, atol = 1e-10))

# ════════════════════════════════════════════════════════════════
# TEST 8: Black-Scholes — expiry guards
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 8: Black-Scholes expiry guards ═══")
itm = bs_all(105.0, 100.0, 0.0, 0.20, 0.05; call = true)
check("ITM call price = intrinsic", itm.price == 5.0)
check("ITM call Δ = 1.0",          itm.Δ == 1.0)
check("Expiry Γ = 0",              itm.Γ == 0.0)

otm = bs_all(95.0, 100.0, 0.0, 0.20, 0.05; call = true)
check("OTM call price = 0", otm.price == 0.0)
check("OTM call Δ = 0.0",   otm.Δ == 0.0)

# ════════════════════════════════════════════════════════════════
# TEST 9: Belief-weighted pricing (Jensen's inequality)
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 9: Belief-weighted pricing ═══")
σ_regimes = [0.121, 0.269]
certain_low  = bs_all_belief_weighted(100.0, 100.0, 0.25, σ_regimes, [1.0, 0.0], 0.05)
direct_low   = bs_all(100.0, 100.0, 0.25, 0.121, 0.05)
certain_high = bs_all_belief_weighted(100.0, 100.0, 0.25, σ_regimes, [0.0, 1.0], 0.05)
direct_high  = bs_all(100.0, 100.0, 0.25, 0.269, 0.05)
mixed        = bs_all_belief_weighted(100.0, 100.0, 0.25, σ_regimes, [0.5, 0.5], 0.05)
σ_mean       = 0.5 * 0.121 + 0.5 * 0.269
price_at_mean = bs_price(100.0, 100.0, 0.25, σ_mean, 0.05)

check("Certain low-vol matches direct",  isapprox(certain_low.price, direct_low.price, atol = 1e-10))
check("Certain high-vol matches direct", isapprox(certain_high.price, direct_high.price, atol = 1e-10))
check("Mixed price between regimes",     direct_low.price < mixed.price < direct_high.price)
check("Jensen's: E[V(σ)] ≥ V(E[σ])",    mixed.price >= price_at_mean - 1e-10)

# ════════════════════════════════════════════════════════════════
# TEST 10: Spot dynamics — constant vol statistics
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 10: Spot dynamics — constant vol ═══")

returns_10, final_vs_10 = let
    rng     = MersenneTwister(123)
    returns = Float64[]
    S       = 100.0
    vs      = VolState(vm_const)
    for _ in 1:100_000
        S, vs, logr = step_spot(S, vs, config, rng)
        push!(returns, logr)
    end
    returns, vs
end

expected_mean = (0.05 - 0.5 * 0.20^2) * (1/252)
expected_std  = 0.20 * sqrt(1/252)
se            = expected_std / sqrt(length(returns_10))
@printf("  Expected mean: %.8f, Actual: %.8f (SE: %.8f)\n",
        expected_mean, mean(returns_10), se)

check("Mean log-return correct (within 3 SE)", abs(mean(returns_10) - expected_mean) < 3 * se)
check("Std log-return correct (within 1%)",
      abs(std(returns_10) - expected_std) / expected_std < 0.01)
check("Regime stays at 1 (constant vol)", final_vs_10.regime_idx == 1)

# ════════════════════════════════════════════════════════════════
# TEST 11: Spot dynamics — regime switching
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 11: Spot dynamics — regime switching ═══")

frac1_11, final_price_11, regime_time_11 = let
    rng         = MersenneTwister(456)
    n_steps     = 500_000
    regime_time = zeros(Int, 2)
    vs          = VolState(vm_hardy)
    S           = 100.0
    for _ in 1:n_steps
        S, vs, _ = step_spot(S, vs, config, rng)
        regime_time[vs.regime_idx] += 1
    end
    regime_time[1] / n_steps, S, regime_time
end

@printf("  Regime 1 time fraction: %.4f (expected ≈ %.4f)\n", frac1_11, expected_π1)
check("Regime 1 time ≈ π₁ (within 2%)", abs(frac1_11 - expected_π1) < 0.02)
check("Both regimes visited",            all(regime_time_11 .> 0))
check("Price stayed positive",           final_price_11 > 0.0)

# ════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════
println("\n" * "═"^50)
println("Results: $passed passed, $failed failed")
println("═"^50)
