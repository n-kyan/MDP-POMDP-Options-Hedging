using Distributions: Normal, pdf, cdf

const STD_NORMAL = Normal(0, 1)

function _d1_d2(S, K, r, q, σ, τ)
    d1 = (log(S/K) + (r - q + 0.5σ^2)*τ) / (σ*√τ)
    d2 = d1 - σ*√τ
    return d1, d2
end

function bs_all(S, σ, contract, τ)

function bs_greeks(
    S::Float64,
    K::Float64,
    r::Float64,
    σ::Float64,
    τ::Float64;
    call::Bool = true
)
    if τ < 1e-8
        return 0.0, 0.0
    end

    d1 = (log(S / K) + (r + 0.5 * σ^2) * τ) / (σ * sqrt(τ))
    d2 = d1 - σ * sqrt(τ)

    Δ = call ? cdf(STD_NORMAL, d1) : cdf(STD_NORMAL, d1) - 1.0
    Γ = pdf(STD_NORMAL, d1) / (S * σ * sqrt(τ))

    return Δ, Γ
end

function bs_price(
    S::Float64,
    K::Float64,
    r::Float64,
    σ::Float64,
    τ::Float64;
    call::Bool = true
)
    if τ < 1e-6
        return call ? max(S - K, 0.0) : max(K -S, 0.0)
    end

    d1 = (log(S / K) + (r + 0.5 * σ^2) * τ) / (σ * sqrt(τ))
    d2 = d1 - σ * sqrt(τ)

    if call
        return S * cdf(STD_NORMAL, d1) - K * exp(-r * τ) * cdf(STD_NORMAL, d2)
    else
        return K * exp(-r * τ) * cdf(STD_NORMAL, -d2) - S * cdf(STD_NORMAL, -d1)
    end
end

function portfolio_price(
    port::Portfolio,
    S::Float64,
    r::Float64,
    σ::Float64,
    t::Int,
    dt::Float64
)
    total = 0.0
    for pos in port.positions
        τ = (pos.T - t) * dt
        total += pos.quantity * bs_price(S, pos.K, r, σ, τ; call=pos.call)

    end
    return total
end

function verify_greeks()
    S, K, r, σ, τ = 100.0, 100.0, 0.05, 0.2, 1.0

    call_Δ, Γ = bs_greeks(S, K, r, σ, τ; call=true)
    put_Δ, _      = bs_greeks(S, K, r, σ, τ; call=false)

    println("=== ATM call (S=K=100, r=5%, σ=20%, τ=1yr) ===")
    println("Call Δ: $(round(call_Δ, digits=4))  (expected ≈ 0.6368)")
    println("Put Δ:  $(round(put_Δ,  digits=4))  (expected ≈ -0.3632)")
    println("Γ:      $(round(Γ,      digits=4))  (expected ≈ 0.0188)")

    println("\n=== Put-call parity check ===")
    println("Call Δ + |Put Δ| = $(round(call_Δ + abs(put_Δ), digits=6))  (should be 1.0)")

    println("\n=== Near-expiry Γ guard ===")
    _, Γ_near = bs_greeks(S, K, r, σ, 1e-7)
    println("Γ at τ=1e-7: $(Γ_near)  (should be 0.0)")
end