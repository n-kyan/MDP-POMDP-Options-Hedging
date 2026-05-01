using Random

# Quotes are placed symmetrically around the agent's believed fair value.
function compute_quotes(hat_V::Float64, δ::Float64)
    return (; bid_price = hat_V - δ, ask_price = hat_V + δ)
end

# AS (2008) fill intensity model.
# λ = A · exp(-k · δ_quote) where δ_quote is the distance from market mid to the quote.
# Clamped: negative δ_quote (quote better than market) → fill probability capped at min(1, A·Δt).
function fill_probability(δ_quote::Float64, A::Float64, k::Float64, Δt::Float64)::Float64
    return min(1.0, A * exp(-k * max(δ_quote, 0.0)) * Δt)
end

# Simulate independent Bernoulli fill events for bid and ask.
#
# Fills are evaluated against V_market (the true consensus price), not hat_V.
# When the agent's belief diverges from V_market, fill probabilities become
# asymmetric — the more favorable side (from the market's perspective) fills
# more often. This asymmetry is the POMDP signal the particle filter uses.
function simulate_fills(
    bid_price::Float64,
    ask_price::Float64,
    V_market::Float64,
    config::SimConfig,
    rng::AbstractRNG
)::FillOutcome
    δ_ask = ask_price - V_market   # positive = ask above market (favorable for agent)
    δ_bid = V_market - bid_price   # positive = bid below market (favorable for agent)

    p_ask = fill_probability(δ_ask, config.A, config.k, config.Δt)
    p_bid = fill_probability(δ_bid, config.A, config.k, config.Δt)

    bid_filled = rand(rng) < p_bid
    ask_filled = rand(rng) < p_ask

    # Scalar fill indicator for the observation vector.
    # +1 = bid only (long inventory added), -1 = ask only (inventory reduced), 0 = no fill or both.
    f_t = (bid_filled && !ask_filled) ? 1 :
          (ask_filled && !bid_filled) ? -1 : 0

    return FillOutcome(bid_filled, ask_filled, bid_price, ask_price, V_market, f_t)
end

# Update portfolio inventory and cash from fill events.
function update_from_fills!(portfolio::Portfolio, fill::FillOutcome, option_idx::Int)
    if fill.bid_filled
        portfolio.option_quantities[option_idx] += 1
        portfolio.cash -= fill.bid_price
    end
    if fill.ask_filled
        portfolio.option_quantities[option_idx] -= 1
        portfolio.cash += fill.ask_price
    end
end
