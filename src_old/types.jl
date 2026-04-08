const HEDGE_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

mutable struct OptionPosition
    K::Float64
    τ::Float64
    is_call::Bool
    quantity::Int
end

struct Portfolio
    positions::Vector{OptionPosition}
end

function portfolio_greeks(
    port::Portfolio,
    S::Float64,
    r::Float64,
    σ::Float64,
    t::Int,
    dt::Float64,
)
    total_Δ = 0.0
    total_Γ = 0.0

    for pos in port.positions
        τ = (pos.T - t) * dt
        Δ, Γ = bs_greeks(S, pos.K, r, σ, τ; call=pos.call)
        total_Δ += pos.quantity * Δ
        total_Γ += pos.quantity * Γ
    end

    return total_Δ, total_Γ
end

function verify_portfolio()
    dt = 1/252
    T  = 63
    S  = 100.0
    r  = 0.05
    σ  = 0.2
    t  = 0

    port_single = Portfolio([
        OptionPosition(100.0, T, true, 1.0)
    ])

    port_multi = Portfolio([
        OptionPosition(100.0, T, true,   1.0),
        OptionPosition(105.0, T, true,  -2.0),
        OptionPosition(95.0,  T, false,  1.0),
    ])

    Δ_s, Γ_s = portfolio_greeks(port_single, S, r, σ, t, dt)
    Δ_single, Γ_single = bs_greeks(S, 100.0, r, σ, T*dt; call=true)

    println("=== Single position matches bs_greeks directly ===")
    println("portfolio_greeks: Δ=$(round(Δ_s, digits=6)), Γ=$(round(Γ_s, digits=6))")
    println("bs_greeks direct: Δ=$(round(Δ_single, digits=6)), Γ=$(round(Γ_single, digits=6))")

    println("\n=== Multi-position portfolio ===")
    Δ_m, Γ_m = portfolio_greeks(port_multi, S, r, σ, t, dt)
    println("Aggregate Δ: $(round(Δ_m, digits=4))")
    println("Aggregate Γ: $(round(Γ_m, digits=4))")

    println("\n=== Expiry behavior: position at t=T ===")
    Δ_exp, Γ_exp = portfolio_greeks(port_single, S, r, σ, T, dt)
    println("Δ at expiry: $(Δ_exp)  (should be 0.0)")
    println("Γ at expiry: $(Γ_exp)  (should be 0.0)")
end