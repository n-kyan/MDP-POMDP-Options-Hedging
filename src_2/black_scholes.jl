using Distributions: Normal, pdf, cdf
include("types.jl")

const STD_NORMAL = Normal(0, 1)


#= 
Inputs:
Option Struct with:
    S - spot price
    K - option strike price (price you can buy the spot at)
    τ - years till expiration
    σ - volatility
    is_call - is this a call option
r - risk free rate of return
q = dividend yield


Calculates:
1. bs_price - Given S, and a european option that lets you buy at strike K in τ years with vol σ, what is the option worth right now?
2. bs_Δ - The amount the price of the option changes when the price of the spot changes by 1.
3. bs_Γ - The rate of change of bs_Δ
=#

# Intermediate Value helper
function 

function bs_price(
    S::Float64,
    σ::Float64,
    o::Option
)
    K = o.K
    τ = o.τ
    q = o.q
    r = o.r
    is_call = o.is_call

    if τ < 1e-9
        return is_call ? max(S - K, 0.0) : max(K - S, 0.0)
    end

    d1 = (log(S / K) + (r - q * σ^2) * τ) / (σ * sqrt(τ))
    d2 = d1 - σ * sqrt(τ)

    if is_call
        return S * cdf(STD_NORMAL, d1) - K * exp(-r * τ) * cdf(STD_NORMAL, d2)
    else
        return K * exp(-r * τ) * cdf(STD_NORMAL, -d2) - S * cdf(STD_NORMAL, -d1)
    end
end

function bs_greeks(
    S::Float64,
    σ::Float64,
    o::Option
)
    K = o.K
    τ = o.τ
    r = o.r
    is_call = o.is_call

    if τ < 1e-6
        return 0.0, 0.0
    end

    d1 = (log(S / K) + (r + 0.5 * σ^2) * τ) / (σ * sqrt(τ))

    Δ = is_call ? cdf(STD_NORMAL, d1) : cdf(STD_NORMAL, d1) - 1.0
    Γ = pdf(STD_NORMAL, d1) / (S * σ * sqrt(τ))

    return Δ, Γ
end

function compute_bs(
    S::Float64,
    σ::Float64,
    o::Option
)
    K = o.K
    τ = o.τ
    is_call = o.is_call
    q = 0
    r = 0.02

    if τ < 1e-9
        bs_price = is_call ? max(S - K, 0.0) : max(K - S, 0.0)
    end

    d1 = (ln(S/K) + r - q + σ²/2 * τ) / (σ*√τ)
    d2 = d1 - σ * √τ

    if is_call
        bs_price = S * exp(-q*τ) * cdf(STD_NORMAL, d1) - K * exp(-r*τ) * cdf(STD_NORMAL, d2)
        bs_Δ = exp(-q*τ) * cdf(STD_NORMAL, d1)
    else
        bs_price = K * exp(-r*τ) * cdf(STD_NORMAL,-d2) - S * exp(-q*τ) * cdf(STD_NORMAL,-d1)
        bs_Δ = exp(-q*τ) * cdf(STD_NORMAL, d1) -1
    end

    bs_Γ = pdf(STD_NORMAL, d1) / (S * σ * sqrt(τ))

    return bs_price, bs_Δ, bs_Γ
end

function test_put_call_parity()
    # have claude write the test
end
