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
    if τ < 1e-8
        return call ? max(S - K, 0.0) : max(K - S, 0.0)
    end

    d1, d2 = _d1_d2(S, K, r, σ, τ)

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
    if τ < 1e-8
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
    
    d1, _ = _d1_d2(S, K, r, σ, τ)

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
    if τ < 1e-8
        return 0.0
    end

    d1, _ = _d1_d2(S, K, r, σ, τ)

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
    if τ < 1e-8
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
    
    d1, d2 = _d1_d2(S, K, r, σ, τ)

    # Price
    if call
        price =  S * cdf(STD_NORMAL, d1) - K * exp(-r * τ) * cdf(STD_NORMAL, d2)
    else
        price =  K * exp(-r * τ) * cdf(STD_NORMAL, -d2) - S * cdf(STD_NORMAL, -d1)
    end

    # Greeks
    Δ = call ? cdf(STD_NORMAL, d1) : cdf(STD_NORMAL, d1) - 1.0
    Γ = pdf(STD_NORMAL, d1) / (S * σ * √τ)
    ν = S * √τ * pdf(STD_NORMAL, d1)

    return (; price, Δ, Γ, ν)
end

function bs_all_belief_weighted(
    S::Float64,
    K::Float64,
    τ::Float64,
    σ_regimes::Vector{Float64},
    regime_beliefs::Vector{Float64}, # needs type - I assume its a Vector{Float64}
    r::Float64;
    call::Bool = true
)
    length(σ_regimes) == length(regime_beliefs) || error("Length of σ_regimes ≠ regime_beliefs")
    
    n = length(σ_regimes)
    # Price
    total_price::Float64 = 0
    total_Δ::Float64 = 0
    total_Γ::Float64 = 0
    total_ν::Float64 = 0
    for i in 1:n
        bs = bs_all(S, K, τ, K, r, call) 

        total_price += regime_beliefs[i] * bs.price
        total_Δ += regime_beliefs[i] * bs.Δ
        total_Γ += regime_beliefs[i] * bs.Γ
        total_ν += regime_beliefs[i] * bs.ν
    end

    return (; total_price, total_Δ, total_Γ, total_ν)
end
