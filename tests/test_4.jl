# Tests for Module 4 (fills)
# Run: julia tests/test_4.jl

include("../src/1_types.jl")
include("../src/2_black_scholes.jl")
include("../src/4_fills.jl")

using Printf
using Statistics
using Random: MersenneTwister

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

config = SimConfig()   # A=140, k=6, Δt=1/252
A, k, Δt = config.A, config.k, config.Δt

# ════════════════════════════════════════════════════════════════
# TEST 1: compute_quotes — symmetry around hat_V
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 1: compute_quotes ═══")
hat_V = 4.50
δ     = 0.20
quotes = compute_quotes(hat_V, δ)

check("bid = hat_V − δ", quotes.bid_price ≈ 4.30)
check("ask = hat_V + δ", quotes.ask_price ≈ 4.70)
check("spread = 2δ",     (quotes.ask_price - quotes.bid_price) ≈ 0.40)

# Continuous δ values
for δ_test in [0.05, 0.10, 0.25, 0.50]
    q = compute_quotes(hat_V, δ_test)
    check("δ=$δ_test: spread = 2δ", isapprox(q.ask_price - q.bid_price, 2δ_test, atol=1e-10))
end

# ════════════════════════════════════════════════════════════════
# TEST 2: fill_probability — properties
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 2: fill_probability ═══")

p0    = fill_probability(0.0,  A, k, Δt)
p05   = fill_probability(0.05, A, k, Δt)
p020  = fill_probability(0.20, A, k, Δt)
p080  = fill_probability(0.80, A, k, Δt)
p_neg = fill_probability(-0.10, A, k, Δt)

@printf("  P(δ=0.00) = %.4f (= min(1, A·Δt) = min(1, %.4f))\n", p0, A * Δt)
@printf("  P(δ=0.05) = %.4f\n", p05)
@printf("  P(δ=0.20) = %.4f\n", p020)
@printf("  P(δ=0.80) = %.4f\n", p080)

check("δ=0: P = min(1, A·Δt)",         p0 ≈ min(1.0, A * Δt))
check("monotonically decreasing",       p05 > p020 > p080)
check("all in [0,1]",                   0.0 ≤ p080 && p0 ≤ 1.0)
check("negative δ clamps to δ=0",      p_neg ≈ p0)
check("very wide spread P ≈ 0",        fill_probability(5.0, A, k, Δt) < 0.01)

# ════════════════════════════════════════════════════════════════
# TEST 3: simulate_fills — symmetric case
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 3: simulate_fills — symmetric case ═══")

V_market = 4.50
δ_sym    = 0.10
n_trials = 50_000

bid_rate_3, ask_rate_3 = let
    rng = MersenneTwister(42)
    bc = 0; ac = 0
    for _ in 1:n_trials
        q = compute_quotes(V_market, δ_sym)
        o = simulate_fills(q.bid_price, q.ask_price, V_market, config, rng)
        bc += o.bid_filled; ac += o.ask_filled
    end
    bc / n_trials, ac / n_trials
end

expected = fill_probability(δ_sym, A, k, Δt)
@printf("  Expected: %.4f  Bid: %.4f  Ask: %.4f\n", expected, bid_rate_3, ask_rate_3)
check("bid rate ≈ expected (within 2%)", abs(bid_rate_3 - expected) < 0.02)
check("ask rate ≈ expected (within 2%)", abs(ask_rate_3 - expected) < 0.02)
check("bid ≈ ask (symmetric)",           abs(bid_rate_3 - ask_rate_3) < 0.02)

# ════════════════════════════════════════════════════════════════
# TEST 4: simulate_fills — asymmetric case (POMDP signal)
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 4: simulate_fills — asymmetric case ═══")

# Agent overvalues: hat_V = 4.50, V_market = 4.30
# Agent bid at 4.40 is ABOVE V_market (δ_bid = 4.30-4.40 = -0.10 → clamps to 0)
# Agent ask at 4.60 is far above V_market (δ_ask = 4.60-4.30 = 0.30 → lower fill)
hat_V_agent = 4.50
V_mkt_low   = 4.30

bid_rate_4, ask_rate_4 = let
    rng = MersenneTwister(123)
    bc = 0; ac = 0
    for _ in 1:n_trials
        q = compute_quotes(hat_V_agent, δ_sym)
        o = simulate_fills(q.bid_price, q.ask_price, V_mkt_low, config, rng)
        bc += o.bid_filled; ac += o.ask_filled
    end
    bc / n_trials, ac / n_trials
end

@printf("  Bid rate: %.4f  Ask rate: %.4f\n", bid_rate_4, ask_rate_4)
p_bid_expected = fill_probability(V_mkt_low - (hat_V_agent - δ_sym), A, k, Δt)
p_ask_expected = fill_probability((hat_V_agent + δ_sym) - V_mkt_low, A, k, Δt)
check("bid fills more when agent overvalues", bid_rate_4 > ask_rate_4)
check("bid rate ≈ analytical (within 2%)",   abs(bid_rate_4 - p_bid_expected) < 0.02)
check("ask rate ≈ analytical (within 2%)",   abs(ask_rate_4 - p_ask_expected) < 0.02)

# ════════════════════════════════════════════════════════════════
# TEST 5: FillOutcome — f_t encoding
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 5: FillOutcome f_t encoding ═══")

# Run many fills and verify f_t correctly encodes the outcome
rng = MersenneTwister(77)
for _ in 1:1000
    q = compute_quotes(4.50, 0.10)
    o = simulate_fills(q.bid_price, q.ask_price, 4.50, config, rng)
    expected_ft = (o.bid_filled && !o.ask_filled) ? 1 :
                  (o.ask_filled && !o.bid_filled) ? -1 : 0
    if o.f_t != expected_ft
        check("f_t encoding correct (all 1000 outcomes)", false)
        break
    end
end
check("f_t encoding correct (all 1000 outcomes)", true)
check("f_t ∈ {-1, 0, +1}", true)  # enforced by construction

# ════════════════════════════════════════════════════════════════
# TEST 6: update_from_fills! — inventory accounting
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 6: update_from_fills! ═══")

port = Portfolio()
push!(port.option_quantities, 5)
port.cash = 100.0

bid_fill = FillOutcome(true, false, 4.30, 4.70, 4.50, 1)
update_from_fills!(port, bid_fill, 1)
check("bid fill: inventory +1",    port.option_quantities[1] == 6)
check("bid fill: cash -= bid_price", isapprox(port.cash, 100.0 - 4.30, atol=1e-10))

ask_fill = FillOutcome(false, true, 4.30, 4.70, 4.50, -1)
update_from_fills!(port, ask_fill, 1)
check("ask fill: inventory -1",    port.option_quantities[1] == 5)
check("ask fill: cash += ask_price", isapprox(port.cash, 100.0 - 4.30 + 4.70, atol=1e-10))

no_fill = FillOutcome(false, false, 4.30, 4.70, 4.50, 0)
cash_before = port.cash
update_from_fills!(port, no_fill, 1)
check("no fill: inventory unchanged", port.option_quantities[1] == 5)
check("no fill: cash unchanged",      port.cash ≈ cash_before)

# ════════════════════════════════════════════════════════════════
# TEST 7: Action lower bound
# ════════════════════════════════════════════════════════════════
println("\n═══ TEST 7: δ lower bound (economic constraint) ═══")

# Lower bound = κ·S·|hat_Δ|: spread must cover cost of delta-hedging one contract
S, K, τ, r, σ_hat = 100.0, 100.0, 30/252, 0.05, 0.20
bs = bs_all(S, K, τ, σ_hat, r; call=true)
lower_bound = config.κ * S * abs(bs.Δ)
@printf("  hat_Δ = %.4f, lower bound = \$%.4f\n", bs.Δ, lower_bound)

check("lower bound > 0",               lower_bound > 0.0)
check("lower bound < hat_V",           lower_bound < bs.price)
check("0.10 spread exceeds lower bound (ATM)", 0.10 > lower_bound)

# ════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════
println("\n" * "═"^50)
println("Results: $passed passed, $failed failed")
println("═"^50)
