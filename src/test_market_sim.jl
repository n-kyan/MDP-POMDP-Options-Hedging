# ============================================================================
# test_market_sim.jl — Tests for the Market Simulation Module
# ============================================================================
# Run with: julia test_market_sim.jl
#
# These tests verify that the simulation produces statistically correct output.
# They don't test every edge case — they test the things that matter for
# producing a correct MDP environment.
# ============================================================================

include("market_sim.jl")

using Statistics
using Printf

# ────────────────────────────────────────────────────────────────────────────
# Test Infrastructure
# ────────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────────
# TEST 1: Struct Construction & Validation
# ────────────────────────────────────────────────────────────────────────────

println("\n═══ TEST 1: Struct Construction ═══")

# Valid constant vol
params_cv = MarketParams(100.0, 0.05, 1/252, ConstantVol(0.20))
check("ConstantVol constructs", params_cv.vol.σ == 0.20)
check("MarketParams fields correct", params_cv.S0 == 100.0 && params_cv.r == 0.05)

# Valid regime switching (3-state, calibrated parameters)
T_mat = [0.977  0.023  0.000;
         0.027  0.967  0.006;
         0.000  0.027  0.973]
vol_rs = RegimeSwitchingVol([0.10, 0.20, 0.40], T_mat, 1)
params_rs = MarketParams(100.0, 0.05, 1/252, vol_rs)
check("RegimeSwitchingVol constructs", length(vol_rs.σ_levels) == 3)

# Invalid constructions should error
try
    ConstantVol(-0.1)
    check("Negative σ rejected", false)
catch
    check("Negative σ rejected", true)
end

try
    bad_T = [0.5 0.3; 0.5 0.5]  # rows don't sum to 1
    RegimeSwitchingVol([0.15, 0.35], bad_T, 1)
    check("Bad transition matrix rejected", false)
catch
    check("Bad transition matrix rejected", true)
end

try
    MarketParams(-100.0, 0.05, 1/252, ConstantVol(0.20))
    check("Negative S0 rejected", false)
catch
    check("Negative S0 rejected", true)
end

# ────────────────────────────────────────────────────────────────────────────
# TEST 2: Single Step (Constant Vol)
# ────────────────────────────────────────────────────────────────────────────

println("\n═══ TEST 2: Single Step — Constant Vol ═══")

rng = MersenneTwister(123)
S = 100.0
vs = init_vol_state(params_cv.vol)

S_new, vs_new = step_price(S, vs, params_cv, rng)
check("Price is positive", S_new > 0.0)
check("Price changed", S_new != S)
check("Vol state unchanged", vs_new isa ConstantVolState)

# Run many single steps and check statistics
# NOTE: This is wrapped in a function to avoid Julia's top-level scoping quirk.
# In Julia, a for loop at the top level of a script treats assignments as new
# local variables if a global with the same name exists. Inside a function,
# all variables are local by default and there's no ambiguity.
function collect_returns(params, n_trials)
    rng = MersenneTwister(456)
    returns = Float64[]
    S_current = 100.0
    vs_current = init_vol_state(params.vol)

    for _ in 1:n_trials
        S_next, vs_current = step_price(S_current, vs_current, params, rng)
        push!(returns, log(S_next / S_current))
        S_current = S_next
    end
    return returns
end

returns = collect_returns(params_cv, 100_000)

# Expected mean of log-returns: (r - σ²/2) × dt
σ = 0.20
dt = 1/252
expected_mean = (0.05 - 0.5 * σ^2) * dt
expected_std = σ * sqrt(dt)
actual_mean = mean(returns)
actual_std = std(returns)

@printf("  Expected mean: %.6f, Actual: %.6f\n", expected_mean, actual_mean)
@printf("  Expected std:  %.6f, Actual: %.6f\n", expected_std, actual_std)

# With 100k samples, the sample mean should be within ~3 standard errors
n_samples = length(returns)
se = expected_std / sqrt(n_samples)
check("Mean log-return correct (within 3 SE)", abs(actual_mean - expected_mean) < 3 * se)
check("Std of log-return correct (within 1%)", abs(actual_std - expected_std) / expected_std < 0.01)

# ────────────────────────────────────────────────────────────────────────────
# TEST 3: Single Step (Regime Switching)
# ────────────────────────────────────────────────────────────────────────────

println("\n═══ TEST 3: Single Step — Regime Switching ═══")

rng = MersenneTwister(789)
S = 100.0
vs_rs = init_vol_state(params_rs.vol)

S_new, vs_new = step_price(S, vs_rs, params_rs, rng)
check("RS price is positive", S_new > 0.0)
check("RS vol state is RegimeVolState", vs_new isa RegimeVolState)
check("RS regime is valid", vs_new.regime in [1, 2, 3])

# ────────────────────────────────────────────────────────────────────────────
# TEST 4: Full Path Simulation
# ────────────────────────────────────────────────────────────────────────────

println("\n═══ TEST 4: Full Path Simulation ═══")

result = simulate_path(params_cv, 252; seed=42, randomize_start=false)
check("Prices length correct", length(result.prices) == 253)         # 252 steps + initial
check("Regimes length correct", length(result.regimes) == 253)
check("Log returns length correct", length(result.log_returns) == 252)
check("First price is S0", result.prices[1] == 100.0)
check("All prices positive", all(p > 0.0 for p in result.prices))
check("All regimes are 1 (constant vol)", all(r == 1 for r in result.regimes))

# Verify log returns match prices
# Wrapped in `let` to avoid Julia's top-level scoping quirk with for loops.
let
    lr_ok = true
    for t in 1:252
        expected_lr = log(result.prices[t + 1] / result.prices[t])
        if !isapprox(result.log_returns[t], expected_lr; atol=1e-12)
            lr_ok = false
            break
        end
    end
    check("Log returns consistent with prices", lr_ok)
end

# ────────────────────────────────────────────────────────────────────────────
# TEST 5: Regime Switching Path — Regime Transitions Work
# ────────────────────────────────────────────────────────────────────────────

println("\n═══ TEST 5: Regime Switching Path ═══")

# Simulate a long path to see regime transitions
result_rs = simulate_path(params_rs, 50_000; seed=42, randomize_start=false)
check("RS path all prices positive", all(p > 0.0 for p in result_rs.prices))

# Check that all three regimes appear
n_regime1 = count(==(1), result_rs.regimes)
n_regime2 = count(==(2), result_rs.regimes)
n_regime3 = count(==(3), result_rs.regimes)
check("All three regimes visited", n_regime1 > 0 && n_regime2 > 0 && n_regime3 > 0)
@printf("  Regime 1 (low vol):    %d steps (%.1f%%)\n", n_regime1, 100 * n_regime1 / 50_001)
@printf("  Regime 2 (medium vol): %d steps (%.1f%%)\n", n_regime2, 100 * n_regime2 / 50_001)
@printf("  Regime 3 (high vol):   %d steps (%.1f%%)\n", n_regime3, 100 * n_regime3 / 50_001)

# Expected stationary distribution computed from eigendecomposition (see stationary_distribution())
# Approximate targets from Cerboni Baiardi (2020): ~60% low, ~33% medium, ~7% high
π_expected = stationary_distribution(vol_rs)
@printf("  Expected stationary: [%.3f, %.3f, %.3f]\n", π_expected[1], π_expected[2], π_expected[3])
actual_fracs = [n_regime1, n_regime2, n_regime3] ./ 50_001
@printf("  Actual fractions:    [%.3f, %.3f, %.3f]\n", actual_fracs[1], actual_fracs[2], actual_fracs[3])
check("Stationary distribution approximately correct (within 3% each regime)",
    all(abs(actual_fracs[i] - π_expected[i]) < 0.03 for i in 1:3))

# Verify no-skip property: low→high and high→low transitions should never occur
# (transition matrix has 0.000 in those entries)
# Wrapped in `let` to avoid Julia's top-level soft-scope warnings on loop variables.
let
    transitions_low_to_high = 0
    transitions_high_to_low = 0
    regimes = result_rs.regimes
    for t in 1:length(regimes)-1
        if regimes[t] == 1 && regimes[t+1] == 3
            transitions_low_to_high += 1
        elseif regimes[t] == 3 && regimes[t+1] == 1
            transitions_high_to_low += 1
        end
    end
    @printf("  Direct low→high transitions: %d (should be 0)\n", transitions_low_to_high)
    @printf("  Direct high→low transitions: %d (should be 0)\n", transitions_high_to_low)
    check("No-skip property: zero direct low↔high transitions",
        transitions_low_to_high == 0 && transitions_high_to_low == 0)
end

# ────────────────────────────────────────────────────────────────────────────
# TEST 5b: Starting Regime Distribution
# ────────────────────────────────────────────────────────────────────────────

println("\n═══ TEST 5b: Starting Regime Distribution ═══")

# TEST 5b uses CALIBRATED_3STATE_VOL's matrix directly — no separate hardcoded fixture.
# This ensures the stationary distribution test is always in sync with the constant.
vol_test = CALIBRATED_3STATE_VOL
π = stationary_distribution(vol_test)
@printf("  Computed stationary distribution: [%.4f, %.4f, %.4f]\n", π[1], π[2], π[3])
check("Stationary π sums to 1", isapprox(sum(π), 1.0; atol=1e-10))
# Target: ~60% low, ~33% medium, ~7% high (Cerboni Baiardi 2020)
# The matrix is derived to reproduce these targets — verify it does so within 3%
check("Stationary π[1] (low) ≈ 0.60 (within 3%)", abs(π[1] - 0.60) < 0.03)
check("Stationary π[2] (medium) ≈ 0.33 (within 3%)", abs(π[2] - 0.33) < 0.03)
check("Stationary π[3] (high) ≈ 0.07 (within 2%)", abs(π[3] - 0.07) < 0.02)
check("All π entries non-negative", all(π .>= 0.0))

# Verify sample_initial_regime produces correct distribution over many samples
function test_regime_sampling(vol, n_samples)
    rng = MersenneTwister(777)
    counts = zeros(Int, length(vol.σ_levels))
    for _ in 1:n_samples
        regime = sample_initial_regime(vol, rng)
        counts[regime] += 1
    end
    return counts ./ n_samples
end

sampled_fracs = test_regime_sampling(vol_test, 50_000)
@printf("  Sampled regime fractions: [%.4f, %.4f, %.4f]\n", sampled_fracs[1], sampled_fracs[2], sampled_fracs[3])
check("Sampled regime fractions ≈ stationary (within 2% each)",
    all(abs(sampled_fracs[i] - π[i]) < 0.02 for i in 1:3))

# Verify with_initial_regime creates correct copy
vol_from_3 = with_initial_regime(vol_test, 3)
check("with_initial_regime sets regime correctly", vol_from_3.initial_regime == 3)
check("with_initial_regime preserves σ_levels", vol_from_3.σ_levels == vol_test.σ_levels)
check("with_initial_regime preserves transition matrix",
    vol_from_3.transition_matrix == vol_test.transition_matrix)

# Verify simulation starting in regime 3 actually starts in regime 3
params_from_3 = MarketParams(100.0, 0.05, 1/252, vol_from_3)
result_from_3 = simulate_path(params_from_3, 100; seed=42, randomize_start=false)
check("Path starting in regime 3 begins in regime 3", result_from_3.regimes[1] == 3)

# ────────────────────────────────────────────────────────────────────────────
# TEST 5c: Randomized Starting Regime
# ────────────────────────────────────────────────────────────────────────────

println("\n═══ TEST 5c: Randomized Starting Regime ═══")

# Run many short episodes with randomize_start=true and check that the
# distribution of starting regimes matches the stationary distribution.
# This is the behavior the MDP agent training loop will use.
function count_start_regimes(params, n_episodes)
    counts = zeros(Int, length(params.vol.σ_levels))
    for ep in 1:n_episodes
        rng = MersenneTwister(ep)   # different seed each episode
        result = simulate_path(params, 10, rng; randomize_start=true)
        counts[result.regimes[1]] += 1
    end
    return counts ./ n_episodes
end

n_episodes = 10_000
start_fracs = count_start_regimes(params_rs, n_episodes)
π_stat = stationary_distribution(vol_rs)
@printf("  Starting regime fractions over %d episodes:\n", n_episodes)
@printf("    Regime 1 (low):    %.3f  (stationary: %.3f)\n", start_fracs[1], π_stat[1])
@printf("    Regime 2 (medium): %.3f  (stationary: %.3f)\n", start_fracs[2], π_stat[2])
@printf("    Regime 3 (high):   %.3f  (stationary: %.3f)\n", start_fracs[3], π_stat[3])
check("Starting regime distribution ≈ stationary (within 2% each)",
    all(abs(start_fracs[i] - π_stat[i]) < 0.02 for i in 1:3))

# Verify randomize_start=false always uses vol.initial_regime
result_fixed = simulate_path(params_rs, 10; seed=1, randomize_start=false)
check("randomize_start=false uses initial_regime from vol struct",
    result_fixed.regimes[1] == params_rs.vol.initial_regime)

# ────────────────────────────────────────────────────────────────────────────
# TEST 6: Reproducibility — Same Seed = Same Path
# ────────────────────────────────────────────────────────────────────────────

println("\n═══ TEST 6: Reproducibility ═══")

result_a = simulate_path(params_cv, 100; seed=99)
result_b = simulate_path(params_cv, 100; seed=99)
check("Same seed → same prices", result_a.prices == result_b.prices)

result_c = simulate_path(params_cv, 100; seed=100)
check("Different seed → different prices", result_a.prices != result_c.prices)

# ────────────────────────────────────────────────────────────────────────────
# TEST 7: Volatility Clustering in Regime-Switching Returns
# ────────────────────────────────────────────────────────────────────────────

println("\n═══ TEST 7: Stylized Facts Check ═══")

# Reuse the long simulation from TEST 5 (50,000 steps)
lr = result_rs.log_returns

# 7a: No autocorrelation in raw returns (should be near zero)
# Lag-1 autocorrelation of raw returns
n_lr = length(lr)
mean_lr = mean(lr)
autocorr_raw = sum((lr[t] - mean_lr) * (lr[t+1] - mean_lr) for t in 1:n_lr-1) /
               sum((lr[t] - mean_lr)^2 for t in 1:n_lr)
@printf("  Autocorrelation of raw returns (lag 1): %.4f\n", autocorr_raw)
check("Raw returns ~uncorrelated (|acf| < 0.02)", abs(autocorr_raw) < 0.02)

# 7b: Positive autocorrelation in absolute returns (volatility clustering)
abs_lr = abs.(lr)
mean_abs = mean(abs_lr)
autocorr_abs = sum((abs_lr[t] - mean_abs) * (abs_lr[t+1] - mean_abs) for t in 1:n_lr-1) /
               sum((abs_lr[t] - mean_abs)^2 for t in 1:n_lr)
@printf("  Autocorrelation of |returns| (lag 1): %.4f\n", autocorr_abs)
check("Absolute returns positively correlated (acf > 0.05)", autocorr_abs > 0.05)

# 7c: Heavy tails (excess kurtosis > 0)
# Normal distribution has kurtosis = 3, so excess kurtosis > 0 means fat tails
mean_lr2 = mean(lr)
var_lr = mean((lr .- mean_lr2).^2)
kurtosis = mean((lr .- mean_lr2).^4) / var_lr^2
excess_kurtosis = kurtosis - 3.0
@printf("  Excess kurtosis: %.2f (>0 = fat tails, normal = 0)\n", excess_kurtosis)
check("Returns have fat tails (excess kurtosis > 0.5)", excess_kurtosis > 0.5)

# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────

println("\n" * "═" ^ 40)
println("Results: $passed passed, $failed failed")
if failed == 0
    println("All tests passed! ✓")
else
    println("Some tests failed — review output above.")
end
println("═" ^ 40)