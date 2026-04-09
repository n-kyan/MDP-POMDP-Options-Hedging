include("../src/1_types.jl")
include("../src/2_black_scholes.jl")
include("../src/4_fills.jl")

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

config = SimConfig()  # A=140, k=6, Δt=1/252

# TEST 1: compute_quotes — basic symmetry
println("\n═══ TEST 1: compute_quotes ═══")

V_believed = 4.50
half_spread = 0.20
quotes = compute_quotes(V_believed, half_spread)

check("bid = V - δ", quotes.bid_price ≈ 4.30)
check("ask = V + δ", quotes.ask_price ≈ 4.70)
check("spread = 2δ", (quotes.ask_price - quotes.bid_price) ≈ 0.40)

# Different spread levels
for (i, δ) in enumerate(config.spread_levels)
    q = compute_quotes(4.50, δ)
    check("spread level $i: spread = $(2δ)", (q.ask_price - q.bid_price) ≈ 2δ)
end

# TEST 2: fill_probability — basic properties
println("\n═══ TEST 2: fill_probability — basic properties ═══")

A, k, Δt = config.A, config.k, config.Δt

# At δ=0 (quoting exactly at market value), probability = min(1, A·Δt)
p_at_market = fill_probability(0.0, A, k, Δt)
@printf("  P(fill | δ=0.00) = %.4f  (= min(1, A·Δt) = min(1, %.4f))\n", p_at_market, A * Δt)
check("δ=0: P = min(1, A·Δt)", p_at_market ≈ min(1.0, A * Δt))

# Monotonically decreasing with δ
p_tight  = fill_probability(0.05, A, k, Δt)
p_medium = fill_probability(0.20, A, k, Δt)
p_wide   = fill_probability(0.80, A, k, Δt)

@printf("  P(fill | δ=0.05) = %.4f\n", p_tight)
@printf("  P(fill | δ=0.20) = %.4f\n", p_medium)
@printf("  P(fill | δ=0.80) = %.4f\n", p_wide)

check("tighter spread → higher fill prob", p_tight > p_medium > p_wide)
check("all probabilities in [0, 1]", 0.0 ≤ p_wide && p_tight ≤ 1.0)

# Negative δ (quoting better than market) should clamp to δ=0
p_negative = fill_probability(-0.10, A, k, Δt)
check("negative δ clamps to δ=0", p_negative ≈ p_at_market)

# Very wide spread → probability near 0
p_very_wide = fill_probability(5.0, A, k, Δt)
check("very wide spread → P ≈ 0", p_very_wide < 0.01)

# TEST 3: fill_probability — across all 5 spread levels
println("\n═══ TEST 3: fill_probability — spread level calibration ═══")

println("  Spread level fill probabilities (per side, per step):")
for (i, δ) in enumerate(config.spread_levels)
    p = fill_probability(δ, A, k, Δt)
    @printf("    Level %d: δ=\$%.2f → P(fill) = %.4f (%.1f%%)\n", i, δ, p, p * 100)
end

# Check that the range spans from meaningful to rare
p_tightest = fill_probability(config.spread_levels[1], A, k, Δt)
p_widest   = fill_probability(config.spread_levels[end], A, k, Δt)
check("tightest level has meaningful fill rate (>10%)", p_tightest > 0.10)
check("widest level has low fill rate (<50%)", p_widest < 0.50)
check("spread across levels > 5:1 ratio", p_tightest / p_widest > 5.0)

# TEST 4: simulate_fills — symmetric case
println("\n═══ TEST 4: simulate_fills — symmetric case ═══")

# When V_believed = V_market, quotes are symmetric around market value
# so bid and ask fill rates should be approximately equal
V_market = 4.50
V_believed = 4.50  # same as market (Level 1 / constant vol)
half_spread = 0.10

rng = MersenneTwister(42)
n_trials = 50_000
bid_count = 0
ask_count = 0

for _ in 1:n_trials
    quotes = compute_quotes(V_believed, half_spread)
    outcome = simulate_fills(quotes.bid_price, quotes.ask_price, V_market, config, rng)
    bid_count += outcome.bid_filled
    ask_count += outcome.ask_filled
end

bid_rate = bid_count / n_trials
ask_rate = ask_count / n_trials
@printf("  Symmetric case: bid rate = %.4f, ask rate = %.4f\n", bid_rate, ask_rate)

# Expected: both should be close to fill_probability(0.10, A, k, Δt)
expected_rate = fill_probability(half_spread, A, k, Δt)
@printf("  Expected rate (analytical): %.4f\n", expected_rate)

check("bid rate ≈ expected (within 2%)", abs(bid_rate - expected_rate) < 0.02)
check("ask rate ≈ expected (within 2%)", abs(ask_rate - expected_rate) < 0.02)
check("bid ≈ ask (symmetric)", abs(bid_rate - ask_rate) < 0.02)

# TEST 5: simulate_fills — asymmetric case (POMDP signal)
println("\n═══ TEST 5: simulate_fills — asymmetric case (POMDP signal) ═══")

# Agent believes option is worth 4.50 but market thinks 4.30
# Agent's ask at 4.60 is only 0.30 above V_market → fills more often
# Agent's bid at 4.40 is only 0.10 below V_market → also fills, but differently
V_believed_agent = 4.50
V_market_shifted = 4.30  # market thinks it's worth less
half_spread = 0.10

rng = MersenneTwister(123)
bid_count = 0
ask_count = 0

for _ in 1:n_trials
    quotes = compute_quotes(V_believed_agent, half_spread)
    outcome = simulate_fills(quotes.bid_price, quotes.ask_price, V_market_shifted, config, rng)
    bid_count += outcome.bid_filled
    ask_count += outcome.ask_filled
end

bid_rate = bid_count / n_trials
ask_rate = ask_count / n_trials
@printf("  Asymmetric case: bid rate = %.4f, ask rate = %.4f\n", bid_rate, ask_rate)

# Agent bid is at 4.40, V_market is 4.30
# δ_bid = V_market - bid = 4.30 - 4.40 = -0.10 → clamps to 0 → high fill rate
# Agent ask is at 4.60, V_market is 4.30  
# δ_ask = ask - V_market = 4.60 - 4.30 = 0.30 → lower fill rate
# So bid should fill MORE than ask when agent overvalues relative to market
check("bid fills more when agent overvalues", bid_rate > ask_rate)

# Verify the direction of the expected analytical values
δ_bid_analytical = V_market_shifted - (V_believed_agent - half_spread)  # 4.30 - 4.40 = -0.10
δ_ask_analytical = (V_believed_agent + half_spread) - V_market_shifted  # 4.60 - 4.30 = 0.30
@printf("  δ_bid = %.2f (clamped to 0), δ_ask = %.2f\n", δ_bid_analytical, δ_ask_analytical)

p_bid_expected = fill_probability(δ_bid_analytical, A, k, Δt)
p_ask_expected = fill_probability(δ_ask_analytical, A, k, Δt)
@printf("  Expected: P(bid) = %.4f, P(ask) = %.4f\n", p_bid_expected, p_ask_expected)

check("simulated bid rate ≈ analytical (within 2%)", abs(bid_rate - p_bid_expected) < 0.02)
check("simulated ask rate ≈ analytical (within 2%)", abs(ask_rate - p_ask_expected) < 0.02)

# TEST 6: FillOutcome struct stores correct data
println("\n═══ TEST 6: FillOutcome — data integrity ═══")

rng = MersenneTwister(999)
quotes = compute_quotes(4.50, 0.20)
outcome = simulate_fills(quotes.bid_price, quotes.ask_price, 4.50, config, rng)

check("bid_price stored", outcome.bid_price ≈ 4.30)
check("ask_price stored", outcome.ask_price ≈ 4.70)
check("V_market stored", outcome.V_market ≈ 4.50)
check("bid_filled is Bool", outcome.bid_filled isa Bool)
check("ask_filled is Bool", outcome.ask_filled isa Bool)

# TEST 7: fill_outcome_likelihood — consistency
println("\n═══ TEST 7: fill_outcome_likelihood ═══")

# Create outcomes for all 4 possibilities and check likelihoods sum to 1
bid_price = 4.30
ask_price = 4.70
V_mkt = 4.50

outcomes = [
    FillOutcome(false, false, bid_price, ask_price, V_mkt),  # no fill
    FillOutcome(true,  false, bid_price, ask_price, V_mkt),  # bid only
    FillOutcome(false, true,  bid_price, ask_price, V_mkt),  # ask only
    FillOutcome(true,  true,  bid_price, ask_price, V_mkt),  # both
]

likelihoods = [fill_outcome_likelihood(o, V_mkt, config) for o in outcomes]
total = sum(likelihoods)

@printf("  Likelihoods: no=%.4f, bid=%.4f, ask=%.4f, both=%.4f\n",
    likelihoods[1], likelihoods[2], likelihoods[3], likelihoods[4])
@printf("  Sum = %.6f\n", total)

check("likelihoods sum to 1", isapprox(total, 1.0, atol=1e-10))
check("all likelihoods ≥ 0", all(l -> l >= 0.0, likelihoods))

# With V_market_j different from the stored V_market, likelihoods change
# (this is how the belief updater distinguishes regimes)
likelihood_low_vol  = fill_outcome_likelihood(outcomes[2], 4.40, config)
likelihood_high_vol = fill_outcome_likelihood(outcomes[2], 4.60, config)
check("different V_market_j → different likelihoods", 
    !isapprox(likelihood_low_vol, likelihood_high_vol, atol=1e-6))

# TEST 8: Integration with Black-Scholes (end-to-end)
println("\n═══ TEST 8: Integration with Black-Scholes ═══")

# Simulate what environment.jl will do: compute V_market and V_believed,
# then feed them to the fill model
S, K, τ, r = 100.0, 100.0, 0.25, 0.05
σ_regimes = [0.121, 0.269]
market_belief = [0.55, 0.45]  # stationary dist
agent_belief  = [0.55, 0.45]  # same in Level 1

V_market  = bs_all_belief_weighted(S, K, τ, σ_regimes, market_belief, r).price
V_believed = bs_all_belief_weighted(S, K, τ, σ_regimes, agent_belief, r).price

@printf("  V_market  = %.4f\n", V_market)
@printf("  V_believed = %.4f\n", V_believed)

quotes = compute_quotes(V_believed, config.spread_levels[2])  # level 2 = $0.10
outcome = simulate_fills(quotes.bid_price, quotes.ask_price, V_market, config, MersenneTwister(42))

@printf("  Quotes: bid=%.4f, ask=%.4f\n", quotes.bid_price, quotes.ask_price)
@printf("  Fill result: bid=%s, ask=%s\n", outcome.bid_filled, outcome.ask_filled)

check("V_market > 0", V_market > 0.0)
check("quotes bracket V_believed", quotes.bid_price < V_believed < quotes.ask_price)
check("end-to-end runs without error", true)

# Summary
println("\n" * "═"^50)
println("Results: $passed passed, $failed failed")
println("═"^50)