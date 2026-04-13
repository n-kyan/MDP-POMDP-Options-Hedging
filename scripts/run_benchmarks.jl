# ============================================================
# tests/test_7.jl — Module 7 Benchmark Tests
# ============================================================
# Run with: julia tests/test_7.jl
# ============================================================

include("../src/7_benchmarks.jl")

using Printf
using Random
using Statistics: mean, std

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

# ── Shared setup ──────────────────────────────────────────────────────────────

config = SimConfig()

# Hardy (2001) two-regime S&P 500 calibration
vm = VolModel(
    [0.10, 0.20],
    transition_matrix = [0.98 0.02; 0.05 0.95]
)

rng = MersenneTwister(42)

# ── TEST 1: GLF-T half-spread formula ─────────────────────────────────────────
println("\n═══ TEST 1: glft_half_spread ═══")

# ATM call: S=100, K=100, τ=63/252 ≈ 0.25 yr, σ=0.20
# Γ for ATM call ≈ N'(d1)/(S·σ·√τ) ≈ 0.3989/(100·0.20·0.5) = 0.040
Γ_atm = bs_all(100.0, 100.0, 63/252, 0.20, 0.05; call=true).Γ
hs_base = glft_half_spread(Γ_atm, 100.0, 0.20, 63/252, config)

@printf("  ATM Γ = %.6f\n", Γ_atm)
@printf("  GLF-T half-spread (σ=0.20, base): \$%.4f\n", hs_base)

check("half-spread is positive", hs_base > 0.0)
check("half-spread is in plausible range [\$0.01, \$1.00]", 0.01 < hs_base < 1.0)

# Higher vol → wider spread (more inventory risk)
hs_highvol = glft_half_spread(Γ_atm, 100.0, 0.40, 63/252, config)
@printf("  GLF-T half-spread (σ=0.40, high vol): \$%.4f\n", hs_highvol)
check("higher vol → wider spread", hs_highvol > hs_base)

# Higher gamma → wider spread
Γ_double = Γ_atm * 2.0
hs_highgamma = glft_half_spread(Γ_double, 100.0, 0.20, 63/252, config)
@printf("  GLF-T half-spread (2×Γ): \$%.4f\n", hs_highgamma)
check("higher gamma → wider spread", hs_highgamma > hs_base)

# Zero gamma → only the market-making term remains (pure spread capture, no inventory risk)
hs_zerogamma = glft_half_spread(0.0, 100.0, 0.20, 63/252, config)
@printf("  GLF-T half-spread (Γ=0): \$%.4f\n", hs_zerogamma)
check("zero gamma → positive spread (market-making term still present)", hs_zerogamma > 0.0)
check("zero gamma spread < base spread", hs_zerogamma < hs_base)

# Negative gamma (short gamma book) → abs(Γ) used, same as positive
hs_neg = glft_half_spread(-Γ_atm, 100.0, 0.20, 63/252, config)
check("negative gamma gives same spread as positive (abs used)", hs_neg ≈ hs_base)

# ── TEST 2: nearest_spread_idx ────────────────────────────────────────────────
println("\n═══ TEST 2: nearest_spread_idx ═══")

# spread_levels = [0.05, 0.10, 0.20, 0.40, 0.80]
check("0.05 → index 1", nearest_spread_idx(0.05, config) == 1)
check("0.10 → index 2", nearest_spread_idx(0.10, config) == 2)
check("0.20 → index 3", nearest_spread_idx(0.20, config) == 3)
check("0.40 → index 4", nearest_spread_idx(0.40, config) == 4)
check("0.80 → index 5", nearest_spread_idx(0.80, config) == 5)

# Midpoints snap to nearest
check("0.07 → index 1 (closer to 0.05)", nearest_spread_idx(0.07, config) == 1)
check("0.08 → index 2 (closer to 0.10)", nearest_spread_idx(0.08, config) == 2)
check("0.13 → index 2 (closer to 0.10)", nearest_spread_idx(0.13, config) == 2)
check("0.29 → index 3 (closer to 0.20 than 0.40)", nearest_spread_idx(0.29, config) == 3)

# Very large value → clamps to last index
check("2.00 → last index (nearest to max spread level)", nearest_spread_idx(2.0, config) == length(config.spread_levels))

# Very small value → clamps to first index
check("0.001 → index 1 (nearest to 0.05)", nearest_spread_idx(0.001, config) == 1)

# ── TEST 3: WW band halfwidth ─────────────────────────────────────────────────
println("\n═══ TEST 3: ww_band_halfwidth ═══")

H_base = ww_band_halfwidth(Γ_atm, 100.0, 0.20, config)
@printf("  WW band halfwidth (base): %.6f Δ-units\n", H_base)

check("WW band is positive", H_base > 0.0)
check("WW band is less than 0.3 (within our Δ_targets grid)", H_base < 0.3)

# Higher gamma → narrower band (need to hedge more often)
H_highgamma = ww_band_halfwidth(Γ_atm * 2.0, 100.0, 0.20, config)
@printf("  WW band halfwidth (2×Γ): %.6f Δ-units\n", H_highgamma)
check("higher gamma → wider band (cubic root of gamma²)", H_highgamma > H_base)
# Note: H ∝ Γ^(2/3), so doubling Γ multiplies H by 2^(2/3) ≈ 1.587, not shrinks it.
# The *relative* width matters: high gamma means even a small delta drift is costly,
# but the formula produces wider H because the cost of being outside is also higher.
# What narrows the band is higher φ (more risk-averse) or lower κ (cheaper to trade).

# Higher transaction cost → wider band (more expensive to rebalance, so tolerate more drift)
config_highκ = SimConfig(κ = 0.01)  # 10× higher transaction cost
H_highκ = ww_band_halfwidth(Γ_atm, 100.0, 0.20, config_highκ)
@printf("  WW band halfwidth (10× κ): %.6f Δ-units\n", H_highκ)
check("higher κ → wider band", H_highκ > H_base)

# Zero gamma → band collapses to near-zero (guard clause kicks in via 1e-10)
H_zerogamma = ww_band_halfwidth(0.0, 100.0, 0.20, config)
@printf("  WW band halfwidth (Γ=0): %.6f Δ-units\n", H_zerogamma)
check("zero gamma → tiny band (gamma floor applied)", H_zerogamma < 0.001)

# ── TEST 4: ww_hedge_idx ──────────────────────────────────────────────────────
println("\n═══ TEST 4: ww_hedge_idx ═══")

# Build a dummy env with known net_Δ values to test the no-trade / trade logic
function make_env_with_delta(net_Δ::Float64, Γ::Float64)
    state = AgentState(
        100.0,       # S
        63/252,      # τ
        net_Δ,       # net_Δ  ← what we're testing
        Γ,           # net_Γ
        0.1,         # net_ν (arbitrary)
        -0.01,       # net_Θ (arbitrary)
        [1.0]        # regime_belief (one regime, certain)
    )
    vm_dummy = VolModel([0.20], transition_matrix = ones(1,1))
    vs_dummy = VolState(vm_dummy, 1)
    return EnvironmentState(
        state,
        vs_dummy,
        [OptionContract(100.0, true)],
        0
    )
end

σ_test = 0.20
H = ww_band_halfwidth(Γ_atm, 100.0, σ_test, config)
@printf("  WW band H = %.6f for test\n", H)

# Inside band → :no_trade (index 1)
env_inside = make_env_with_delta(H * 0.5, Γ_atm)
idx_inside = ww_hedge_idx(env_inside, config, σ_test)
@printf("  net_Δ = %.6f (inside band) → hedge_idx = %d\n", H * 0.5, idx_inside)
check("inside band → :no_trade (index 1)", idx_inside == 1)

# Outside band, positive side → trade toward positive band edge
env_outside_pos = make_env_with_delta(0.25, Γ_atm)
idx_outside_pos = ww_hedge_idx(env_outside_pos, config, σ_test)
@printf("  net_Δ = 0.25 (outside band, positive) → hedge_idx = %d (target = %s)\n",
        idx_outside_pos, string(config.Δ_targets[idx_outside_pos]))
check("outside band positive → trades (not :no_trade)", idx_outside_pos != 1)
check("outside band positive → targets positive side (reduces delta toward zero)",
      config.Δ_targets[idx_outside_pos] isa Float64 &&
      Float64(config.Δ_targets[idx_outside_pos]) >= 0.0)

# Outside band, negative side → trade toward negative band edge
env_outside_neg = make_env_with_delta(-0.25, Γ_atm)
idx_outside_neg = ww_hedge_idx(env_outside_neg, config, σ_test)
@printf("  net_Δ = -0.25 (outside band, negative) → hedge_idx = %d (target = %s)\n",
        idx_outside_neg, string(config.Δ_targets[idx_outside_neg]))
check("outside band negative → trades (not :no_trade)", idx_outside_neg != 1)
check("outside band negative → targets negative side",
      config.Δ_targets[idx_outside_neg] isa Float64 &&
      Float64(config.Δ_targets[idx_outside_neg]) <= 0.0)

# Zero net_Δ → always :no_trade (at center of band)
env_zero = make_env_with_delta(0.0, Γ_atm)
check("net_Δ = 0 → :no_trade", ww_hedge_idx(env_zero, config, σ_test) == 1)

# ── TEST 5: naive_hedge_idx ───────────────────────────────────────────────────
println("\n═══ TEST 5: naive_hedge_idx ═══")

idx_naive = naive_hedge_idx(config)
@printf("  naive_hedge_idx = %d (target = %s)\n", idx_naive, string(config.Δ_targets[idx_naive]))
check("naive hedge targets 0.0", config.Δ_targets[idx_naive] == 0.0)
check("naive hedge is never :no_trade", idx_naive != 1)

# ── TEST 6: symmetric_spread_idx ─────────────────────────────────────────────
println("\n═══ TEST 6: symmetric_spread_idx ═══")

check("default level 2", symmetric_spread_idx() == 2)
check("explicit level 3", symmetric_spread_idx(3) == 3)

# ── TEST 7: Combined policies return valid MarketMakingAction ─────────────────
println("\n═══ TEST 7: combined policy actions ═══")



# Build a proper environment
env_full = EnvironmentState(
    AgentState(100.0, 63/252, 0.05, Γ_atm, 10.0, -0.02, [1.0, 0.0]),
    VolState(vm),
    [OptionContract(100.0, true)],
    0
)
port_full = Portfolio()
push!(port_full.option_quantities, 1)
port_full.q_spot = -0.5
port_full.cash   = 50.0

σ_oracle = get_σ(env_full.vol_state)

action_glft_ww = glft_ww_policy(env_full, port_full, config, σ_oracle)
@printf("  glft_ww_policy → spread_idx=%d (δ=\$%.2f), hedge_idx=%d (target=%s)\n",
        action_glft_ww.spread_idx,
        config.spread_levels[action_glft_ww.spread_idx],
        action_glft_ww.hedge_idx,
        string(config.Δ_targets[action_glft_ww.hedge_idx]))
check("glft_ww spread_idx in range", 1 <= action_glft_ww.spread_idx <= length(config.spread_levels))
check("glft_ww hedge_idx in range",  1 <= action_glft_ww.hedge_idx  <= length(config.Δ_targets))

action_sym_naive = symmetric_naive_policy(env_full, port_full, config, σ_oracle)
@printf("  symmetric_naive_policy → spread_idx=%d, hedge_idx=%d\n",
        action_sym_naive.spread_idx, action_sym_naive.hedge_idx)
check("symmetric_naive spread = level 2", action_sym_naive.spread_idx == 2)
check("symmetric_naive hedge targets 0.0", config.Δ_targets[action_sym_naive.hedge_idx] == 0.0)

# ── TEST 8: run_benchmark smoke test ─────────────────────────────────────────
println("\n═══ TEST 8: run_benchmark — smoke test ═══")

# Use constant vol (level 1) to avoid any belief update complexity in this test
vm_const = VolModel([0.20], transition_matrix = ones(1,1))
config_l1 = SimConfig()
rng_test = MersenneTwister(99)

# Oracle σ_fn for constant vol is just the single regime
σ_fn_const = env -> get_σ(env.vol_state)

# Run a short benchmark (10 episodes, level=1) — this exercises the full loop
results = run_benchmark(
    glft_ww_policy,
    σ_fn_const,
    vm_const,
    config_l1,
    10,
    rng_test;
    level = 1
)

@printf("  GLF-T+WW (constant vol, 10 episodes):\n")
@printf("    Mean P&L per episode: \$%.4f\n", mean(results.episode_pnl))
@printf("    Sharpe:               %.4f\n",  results.sharpe)
@printf("    Mean spread index:    %.2f\n",  results.mean_spread_idx)
@printf("    Hedge frequency:      %.2f%%\n", results.hedge_freq * 100)
@printf("    Mean |net_Δ|:         %.4f\n",  results.mean_abs_net_Δ)

check("returns 10 episode P&Ls", length(results.episode_pnl) == 10)
check("all P&Ls are finite", all(isfinite, results.episode_pnl))
check("Sharpe is finite", isfinite(results.sharpe))
check("mean_spread_idx in valid range", 
      1.0 <= results.mean_spread_idx <= length(config.spread_levels))
check("hedge_freq in [0, 1]", 0.0 <= results.hedge_freq <= 1.0)
check("mean_abs_net_Δ ≥ 0", results.mean_abs_net_Δ >= 0.0)

# ── TEST 9: oracle vs constant-vol produce different behavior (on regime-switching vol) ──
println("\n═══ TEST 9: oracle σ vs constant σ differ on regime-switching model ═══")

rng_a = MersenneTwister(7)
rng_b = MersenneTwister(7)  # identical seed so only σ_fn differs

σ_fn_oracle   = env -> get_σ(env.vol_state)
σ_fn_constant = env -> 0.15  # fixed at low-regime vol

results_oracle = run_benchmark(glft_ww_policy, σ_fn_oracle,   vm, config, 20, rng_a; level = 2)
results_const  = run_benchmark(glft_ww_policy, σ_fn_constant, vm, config, 20, rng_b; level = 2)

@printf("  Oracle   σ: mean spread_idx = %.2f, hedge_freq = %.2f%%\n",
        results_oracle.mean_spread_idx, results_oracle.hedge_freq * 100)
@printf("  Constant σ: mean spread_idx = %.2f, hedge_freq = %.2f%%\n",
        results_const.mean_spread_idx,  results_const.hedge_freq * 100)

check("oracle and constant-vol produce different hedge frequencies",
      abs(results_oracle.hedge_freq - results_const.hedge_freq) > 0.005)

# ── Summary ───────────────────────────────────────────────────────────────────
println("\n══════════════════════════════════════════")
println("Results: $passed passed, $failed failed")
if failed == 0
    println("All tests passed ✓")
else
    println("$failed test(s) FAILED ✗")
end
println("══════════════════════════════════════════")