using Distributions: Normal, pdf, cdf

const STD_NORMAL = Normal(0, 1)

function _d1_d2(
    S::Float64,
    K::Float64,
    τ::Float64,
    σ::Float64,
    r::Float64
)
    d1 = (log(S / K) + (r + 0.5 * σ^2) * τ) / (σ * √τ)
    d2 = d1 - σ * √τ

    return d1, d2
end

function bs_price(
    S::Float64,
    K::Float64,
    τ::Float64,
    σ::Float64,
    r::Float64;
    call::Bool = true
)
    if τ ≈ 0
        return call ? max(S - K, 0.0) : max(K - S, 0.0)
    end

    d1, d2 = _d1_d2(S, K, τ, σ, r)

    if call
        return S * cdf(STD_NORMAL, d1) - K * exp(-r * τ) * cdf(STD_NORMAL, d2)
    else
        return K * exp(-r * τ) * cdf(STD_NORMAL, -d2) - S * cdf(STD_NORMAL, -d1)
    end
end

function bs_Δ_Γ(
    S::Float64,
    K::Float64,
    τ::Float64,
    σ::Float64,
    r::Float64;
    call::Bool = true
)
    if τ ≈ 0
        Γ = 0.0

        if S > K # itm call or otm put
            Δ = call ? 1.0 : 0.0
        elseif S < K # otm call or itm put
            Δ = call ? 0.0 : -1.0
        else # exactly atm
            # Using 0.5 to represent 50/50 probability
            # Mention epsilon buffer ("moneyness smoothing") as alt in paper TODO:
            Δ = call ? 0.5 : -0.5
        end
        return (; Δ, Γ)
    end
    
    d1, _ = _d1_d2(S, K, τ, σ, r)

    Δ = call ? cdf(STD_NORMAL, d1) : cdf(STD_NORMAL, d1) - 1.0
    Γ = pdf(STD_NORMAL, d1) / (S * σ * √τ)

    return (; Δ, Γ)
end

function bs_ν(
    S::Float64,
    K::Float64,
    τ::Float64,
    σ::Float64,
    r::Float64;
)
    if τ ≈ 0
        return 0.0
    end

    d1, _ = _d1_d2(S, K, τ, σ, r)
    return S * √τ * pdf(STD_NORMAL, d1)
end

function bs_all(
    S::Float64,
    K::Float64,
    τ::Float64,
    σ::Float64,
    r::Float64;
    call::Bool = true
)
    if τ ≈ 0
        price = call ? max(S - K, 0.0) : max(K - S, 0.0)
        Γ = 0.0
        ν = 0.0

        if S > K # itm call or otm put
            Δ = call ? 1.0 : 0.0
        elseif S < K # otm call or itm put
            Δ = call ? 0.0 : -1.0
        else # exactly atm
            # Using 0.5 to represent 50/50 probability
            # Mention epsilon buffer ("moneyness smoothing") as alt in paper TODO:
            Δ = call ? 0.5 : -0.5
        end

        return (; price, Δ, Γ, ν)
    end
    
    # Normal Case
    d1, d2 = _d1_d2(S, K, τ, σ, r)
    pdf_d1 = pdf(STD_NORMAL, d1)
    cdf_d1 = cdf(STD_NORMAL, d1)

    # Price
    if call
        price =  S * cdf_d1 - K * exp(-r * τ) * cdf(STD_NORMAL, d2)
    else
        price =  K * exp(-r * τ) * cdf(STD_NORMAL, -d2) - S * cdf(STD_NORMAL, -d1)
    end

    # Greeks
    Δ = call ? cdf_d1 : cdf_d1 - 1.0
    Γ = pdf_d1 / (S * σ * √τ)
    ν = S * √τ * pdf_d1

    return (; price, Δ, Γ, ν)
end

#=
Compute belief-weighted price and Greeks:
    V = Σᵢ beliefs[i] × V_BS(σ_regimes[i])

This is mathematically correct because BS pricing is nonlinear in σ
(Jensen's inequality), so you must price per-regime and then average,
not average σ and then price.

Used by both the agent (beliefs = agent's belief) and the market
(beliefs = market's returns-only filter belief).
=#

function bs_all_belief_weighted(
    S::Float64,
    K::Float64,
    τ::Float64,
    σ_regimes::Vector{Float64},
    regime_beliefs::Vector{Float64},
    r::Float64;
    call::Bool = true
)
    length(σ_regimes) == length(regime_beliefs) || error(
        "σ_regimes has $(length(σ_regimes)) entries but beliefs has $(length(regime_beliefs))"
        )
    
    n = length(σ_regimes)
    # Price
    total_price = 0.0
    total_Δ = 0.0
    total_Γ = 0.0
    total_ν = 0.0

    for i in 1:n
        bs_i = bs_all(S, K, τ, σ_regimes[i], r; call=call) 

        total_price += regime_beliefs[i] * bs_i.price
        total_Δ += regime_beliefs[i] * bs_i.Δ
        total_Γ += regime_beliefs[i] * bs_i.Γ
        total_ν += regime_beliefs[i] * bs_i.ν
    end

    return (price=total_price, Δ=total_Δ, Γ=total_Γ, ν=total_ν)
end
