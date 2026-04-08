using Distributions: Normal, pdf, cdf

# ============================================================
# TYPES
# (In your project, OptionContract lives in types.jl and is
#  pulled in via include("types.jl"). It's shown here for
#  reference so this file is self-contained.)
# ============================================================

# struct OptionContract
#     K::Float64       # Strike price
#     T::Float64       # Total life of the option in years (fixed at creation)
#     r::Float64       # Risk-free rate, annualized (e.g. 0.05 = 5%)
#     q::Float64       # Dividend yield (use 0.0 for non-dividend-paying assets)
#     is_call::Bool    # true = call option, false = put option
# end
#
# NOTE on T vs τ:
#   T  — total option life, stored in the contract, never changes.
#   τ  — time REMAINING until expiry, passed as an argument at each step.
#        τ starts at T and counts down to 0 as the simulation advances.
#        Always use τ (not T) inside these functions.


# ============================================================
# CONSTANTS
# ============================================================

const STD_NORMAL = Normal(0.0, 1.0)


# ============================================================
# PRIVATE HELPERS
# ============================================================

# _d1_d2: Computes the two intermediate scalars that appear in every
# Black-Scholes formula. Extracted here so we never recompute them twice
# in the same call path.
#
# Returns: (d1, d2) as a tuple of Float64.
#
# Formula:
#   d1 = [ln(S/K) + (r - q + σ²/2) * τ] / (σ * √τ)
#   d2 = d1 - σ * √τ
function _d1_d2(S::Float64, K::Float64, r::Float64, q::Float64,
                σ::Float64, τ::Float64)::Tuple{Float64, Float64}
    sqrt_τ = sqrt(τ)
    d1 = (log(S / K) + (r - q + 0.5 * σ^2) * τ) / (σ * sqrt_τ)
    d2 = d1 - σ * sqrt_τ
    return d1, d2
end


# ============================================================
# PRICE
# ============================================================

# bs_price: Black-Scholes theoretical value of a European option.
#
# Args:
#   S  — current spot price of the underlying
#   σ  — volatility, annualized (e.g. 0.20 = 20%)
#   o  — OptionContract (holds K, r, q, is_call)
#   τ  — time remaining to expiry in years (e.g. 30 days = 30/252)
#
# Returns: option price as Float64.
#
# At expiry (τ ≈ 0), returns the intrinsic value (payoff at expiry).
function bs_price(S::Float64, σ::Float64, o::OptionContract, τ::Float64)::Float64
    # --- Handle expiry ---
    # When τ is effectively zero, the option is worth only its intrinsic value.
    # We must guard here because d1/d2 divide by σ*√τ, which → 0.
    if τ < 1e-9
        return o.is_call ? max(S - o.K, 0.0) : max(o.K - S, 0.0)
    end

    d1, d2 = _d1_d2(S, o.K, o.r, o.q, σ, τ)

    if o.is_call
        # Call: S * exp(-qτ) * N(d1)  -  K * exp(-rτ) * N(d2)
        return S * exp(-o.q * τ) * cdf(STD_NORMAL, d1) -
               o.K * exp(-o.r * τ) * cdf(STD_NORMAL, d2)
    else
        # Put: K * exp(-rτ) * N(-d2)  -  S * exp(-qτ) * N(-d1)
        return o.K * exp(-o.r * τ) * cdf(STD_NORMAL, -d2) -
               S * exp(-o.q * τ) * cdf(STD_NORMAL, -d1)
    end
end


# ============================================================
# GREEKS — Individual functions
# ============================================================
# Each function is a thin wrapper around _d1_d2. Keeping them
# separate lets you call only what you need, which is cleaner
# at call sites like:
#
#   δ = bs_delta(S, σ, contract, τ)
#
# rather than always unpacking a tuple.

# bs_delta: Sensitivity of option price to a $1 move in the underlying.
#
# Range: [0, 1] for calls, [-1, 0] for puts.
# Intuition: an ATM call has delta ≈ 0.5 — the option gains $0.50 for
# every $1 the stock gains. Your hedging agent uses this to decide how
# many shares of underlying to hold.
function bs_delta(S::Float64, σ::Float64, o::OptionContract, τ::Float64)::Float64
    if τ < 1e-9
        if o.is_call
            return S > o.K ? 1.0 : 0.0
        else
            return S < o.K ? -1.0 : 0.0
        end
    end

    d1, _ = _d1_d2(S, o.K, o.r, o.q, σ, τ)

    if o.is_call
        return exp(-o.q * τ) * cdf(STD_NORMAL, d1)
    else
        # Put delta = call delta - 1  (put-call parity relationship)
        return exp(-o.q * τ) * (cdf(STD_NORMAL, d1) - 1.0)
    end
end


# bs_gamma: Rate of change of delta with respect to the underlying price.
#
# Gamma is always POSITIVE for both calls and puts (long options).
# It peaks when the option is at-the-money and near expiry — this is
# when delta is most sensitive, making hedging most expensive and urgent.
#
# Key insight for your project: high gamma → the agent should hedge
# more aggressively because delta is about to shift a lot.
function bs_gamma(S::Float64, σ::Float64, o::OptionContract, τ::Float64)::Float64
    if τ < 1e-9
        return 0.0
    end

    d1, _ = _d1_d2(S, o.K, o.r, o.q, σ, τ)

    # Gamma = N'(d1) / (S * σ * √τ)
    # N'(d1) is the standard normal PDF evaluated at d1.
    return exp(-o.q * τ) * pdf(STD_NORMAL, d1) / (S * σ * sqrt(τ))
end


# bs_vega: Sensitivity of option price to a 1-unit change in volatility σ.
#
# Vega is always POSITIVE for both calls and puts.
# Intuition: higher vol → wider distribution of possible outcomes →
# option is worth more (optionality is more valuable).
#
# In your paper, you can use vega to show the scale of vol risk in your
# simulated positions — this is directly relevant to the Level 3 story
# about the cost of not knowing the true volatility regime.
#
# Note: conventionally quoted per 1% move in vol (divide by 100 at call sites
# if you want that convention). Here we return raw sensitivity per unit of σ.
function bs_vega(S::Float64, σ::Float64, o::OptionContract, τ::Float64)::Float64
    if τ < 1e-9
        return 0.0
    end

    d1, _ = _d1_d2(S, o.K, o.r, o.q, σ, τ)

    # Vega = S * exp(-qτ) * N'(d1) * √τ
    # Same for calls and puts.
    return S * exp(-o.q * τ) * pdf(STD_NORMAL, d1) * sqrt(τ)
end


# ============================================================
# COMBINED FUNCTION
# ============================================================
# bs_all: Compute price and all Greeks in one call, reusing d1/d2.
#
# Use this in your reward function and anywhere you need more than one
# output at the same (S, σ, τ). Returns a NamedTuple so call sites are
# readable:
#
#   result = bs_all(100.0, 0.2, contract, 0.25)
#   result.price   # → Float64
#   result.delta   # → Float64
#   result.gamma   # → Float64
#   result.vega    # → Float64
function bs_all(S::Float64, σ::Float64, o::OptionContract, τ::Float64)
    # --- Handle expiry ---
    if τ < 1e-9
        price = o.is_call ? max(S - o.K, 0.0) : max(o.K - S, 0.0)
        delta = o.is_call ? (S > o.K ? 1.0 : 0.0) : (S < o.K ? -1.0 : 0.0)
        return (price=price, delta=delta, gamma=0.0, vega=0.0)
    end

    d1, d2 = _d1_d2(S, o.K, o.r, o.q, σ, τ)
    sqrt_τ  = sqrt(τ)
    exp_qτ  = exp(-o.q * τ)
    exp_rτ  = exp(-o.r * τ)
    N_d1    = cdf(STD_NORMAL, d1)
    N_d2    = cdf(STD_NORMAL, d2)
    N_neg_d1 = 1.0 - N_d1   # cdf(STD_NORMAL, -d1) = 1 - N(d1)
    N_neg_d2 = 1.0 - N_d2
    n_d1    = pdf(STD_NORMAL, d1)   # Standard normal PDF at d1

    if o.is_call
        price = S * exp_qτ * N_d1  - o.K * exp_rτ * N_d2
        delta = exp_qτ * N_d1
    else
        price = o.K * exp_rτ * N_neg_d2 - S * exp_qτ * N_neg_d1
        delta = exp_qτ * (N_d1 - 1.0)
    end

    gamma = exp_qτ * n_d1 / (S * σ * sqrt_τ)
    vega  = S * exp_qτ * n_d1 * sqrt_τ

    return (price=price, delta=delta, gamma=gamma, vega=vega)
end


# ============================================================
# TESTS
# ============================================================
# Run with: include("black_scholes.jl"); test_black_scholes()
# All tests print PASS or FAIL with the computed vs expected values.

function test_black_scholes()
    println("=" ^ 50)
    println("Running Black-Scholes tests...")
    println("=" ^ 50)

    # --- Shared test contract ---
    # ATM call, 3 months to expiry, 20% vol, 5% risk-free rate
    atm_call = OptionContract(100.0, 0.25, 0.05, 0.0, true)
    atm_put  = OptionContract(100.0, 0.25, 0.05, 0.0, false)
    S = 100.0
    σ = 0.20
    τ = 0.25

    # --------------------------------------------------
    # TEST 1.1: Put-Call Parity
    # call - put = S*exp(-qτ) - K*exp(-rτ)
    # Must hold to machine precision (tolerance 1e-10).
    # --------------------------------------------------
    call_price = bs_price(S, σ, atm_call, τ)
    put_price  = bs_price(S, σ, atm_put,  τ)
    lhs = call_price - put_price
    rhs = S * exp(-atm_call.q * τ) - atm_call.K * exp(-atm_call.r * τ)
    parity_error = abs(lhs - rhs)
    passed = parity_error < 1e-10
    println("TEST 1.1 Put-Call Parity:  $(passed ? "PASS" : "FAIL")")
    println("  call=$call_price, put=$put_price, error=$parity_error")

    # Also check several other (S, K, σ, τ) combinations
    for (S2, K2, σ2, τ2) in [(90.0, 100.0, 0.3, 0.5), (110.0, 100.0, 0.15, 0.1),
                               (100.0, 95.0, 0.25, 1.0)]
        c = OptionContract(K2, τ2, 0.05, 0.0, true)
        p = OptionContract(K2, τ2, 0.05, 0.0, false)
        err = abs((bs_price(S2, σ2, c, τ2) - bs_price(S2, σ2, p, τ2)) -
                  (S2 * exp(-c.q * τ2) - K2 * exp(-c.r * τ2)))
        ok = err < 1e-10
        println("  Additional parity check (S=$S2, K=$K2, σ=$σ2, τ=$τ2): $(ok ? "PASS" : "FAIL") err=$err")
    end

    # --------------------------------------------------
    # TEST 1.2: Delta Bounds
    # Call delta ∈ [0, 1]. Put delta ∈ [-1, 0].
    # Check across a range of moneyness levels.
    # --------------------------------------------------
    println("\nTEST 1.2 Delta Bounds:")
    all_pass = true
    for S_test in [70.0, 85.0, 100.0, 115.0, 130.0]
        c = OptionContract(100.0, 0.25, 0.05, 0.0, true)
        p = OptionContract(100.0, 0.25, 0.05, 0.0, false)
        δ_call = bs_delta(S_test, σ, c, τ)
        δ_put  = bs_delta(S_test, σ, p, τ)
        ok = (0.0 ≤ δ_call ≤ 1.0) && (-1.0 ≤ δ_put ≤ 0.0)
        all_pass = all_pass && ok
        println("  S=$S_test → call_delta=$δ_call, put_delta=$δ_put  $(ok ? "PASS" : "FAIL")")
    end

    # --------------------------------------------------
    # TEST 1.3: Gamma is Positive (same for calls and puts)
    # --------------------------------------------------
    println("\nTEST 1.3 Gamma Positive & Call/Put Equal:")
    for S_test in [80.0, 100.0, 120.0]
        c = OptionContract(100.0, 0.25, 0.05, 0.0, true)
        p = OptionContract(100.0, 0.25, 0.05, 0.0, false)
        Γ_call = bs_gamma(S_test, σ, c, τ)
        Γ_put  = bs_gamma(S_test, σ, p, τ)
        ok = Γ_call > 0.0 && abs(Γ_call - Γ_put) < 1e-12
        println("  S=$S_test → gamma_call=$Γ_call, gamma_put=$Γ_put  $(ok ? "PASS" : "FAIL")")
    end

    # --------------------------------------------------
    # TEST 1.4: ATM Delta ≈ 0.5
    # A call with S=K, moderate vol, and a few months to expiry
    # should have delta near 0.5 (within 0.05).
    # --------------------------------------------------
    atm_delta = bs_delta(100.0, 0.20, atm_call, 0.25)
    passed = abs(atm_delta - 0.5) < 0.05
    println("\nTEST 1.4 ATM Delta ≈ 0.5:  $(passed ? "PASS" : "FAIL")")
    println("  ATM call delta = $atm_delta  (expected ≈ 0.5)")

    # --------------------------------------------------
    # TEST 1.5: Expiry Limits
    # At τ=0, ITM call → delta=1, OTM call → delta=0.
    # At τ=0, price = intrinsic value.
    # --------------------------------------------------
    println("\nTEST 1.5 Expiry Limits:")
    itm_call = OptionContract(90.0, 0.0, 0.05, 0.0, true)   # S=100 > K=90 → ITM
    otm_call = OptionContract(110.0, 0.0, 0.05, 0.0, true)  # S=100 < K=110 → OTM

    itm_price = bs_price(100.0, 0.20, itm_call, 0.0)
    otm_price = bs_price(100.0, 0.20, otm_call, 0.0)
    itm_delta = bs_delta(100.0, 0.20, itm_call, 0.0)
    otm_delta = bs_delta(100.0, 0.20, otm_call, 0.0)

    println("  ITM call at expiry → price=$(itm_price) (expect 10.0) $(abs(itm_price - 10.0) < 1e-10 ? "PASS" : "FAIL")")
    println("  OTM call at expiry → price=$(otm_price) (expect 0.0)  $(abs(otm_price - 0.0) < 1e-10 ? "PASS" : "FAIL")")
    println("  ITM call at expiry → delta=$(itm_delta) (expect 1.0)  $(itm_delta == 1.0 ? "PASS" : "FAIL")")
    println("  OTM call at expiry → delta=$(otm_delta) (expect 0.0)  $(otm_delta == 0.0 ? "PASS" : "FAIL")")

    # --------------------------------------------------
    # TEST 1.6: bs_all consistency
    # bs_all must agree with individual functions to machine precision.
    # --------------------------------------------------
    println("\nTEST 1.6 bs_all Consistency:")
    result = bs_all(S, σ, atm_call, τ)
    price_ok = abs(result.price - bs_price(S, σ, atm_call, τ)) < 1e-12
    delta_ok = abs(result.delta - bs_delta(S, σ, atm_call, τ)) < 1e-12
    gamma_ok = abs(result.gamma - bs_gamma(S, σ, atm_call, τ)) < 1e-12
    vega_ok  = abs(result.vega  - bs_vega(S, σ, atm_call, τ))  < 1e-12
    all_ok = price_ok && delta_ok && gamma_ok && vega_ok
    println("  price=$(price_ok ? "PASS" : "FAIL"), delta=$(delta_ok ? "PASS" : "FAIL"), gamma=$(gamma_ok ? "PASS" : "FAIL"), vega=$(vega_ok ? "PASS" : "FAIL")")
    println("  bs_all overall: $(all_ok ? "PASS" : "FAIL")")

    println("\n" * "=" ^ 50)
    println("Tests complete.")
end