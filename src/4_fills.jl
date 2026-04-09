using Random

# Record of what happended at each timestep
struct FillOutcome
    bid_filled::Bool
    ask_filled::Bool
    bid_price::Float64
    ask_price::Float64
    V_market::Float64
end

#=
- Compute the agent's bid and ask centered around its believed fair value which comes
from bs_all_belief_weighted().

- The half_spread comes from the agent's decision from the SimConfig. The agent must quote symetrically.

- If the agent's belief diverges from the market's belief, implying a different believed fair value, the market will produce asymmetric fill probabilities. This aymmetry is a POMDP observation signal.
=#
function compute_quotes(V_believed::Float64, half_spread::Float64)
    bid_price = V_believed - half_spread
    ask_price = V_believed + half_spread
    return (; bid_price, ask_price)
end

# Fill probabilities from Avellaneda & Stoikov (2008) exponential fill intensity model
function fill_probability( # TODO: add comments explaining each parameter and how the math works and why it is the correct model
    δ::Float64,
    A::Float64,
    k::Float64,
    Δt::Float64
    )
    # Negative δ means our quote is better than the market price (offering to buy above market or sell below market). So limit δ at 0 so we get P = min(1, A·Δt).
    δ_clamped = max(δ, 0.0)
    return min(1.0, A * exp(-k * δ_clamped) * Δt)
end

#= 
Simulates if bids and asks get filled at each timestep.

Probabilities are computed against market consensus price not against the agent's price. This is what makes fill asymmetry informative for the agent.
=#
function simulate_fills(
    bid_price::Float64,
    ask_price::Float64,
    V_market::Float64,
    config::SimConfig,
    rng::AbstractRNG
)
    A = config.A
    k = config.k
    Δt = config.Δt

    # Distance from market value to each quote
    # Ask is above V_market (positive δ = someone must pay a premium to buy from us)
    # Bid is below V_market (positive δ = someone must accept a discount to sell to us)
    δ_ask = ask_price - V_market
    δ_bid = V_market - bid_price

    # Fill probabilities
    p_ask = fill_probability(δ_ask, A, k, Δt)
    p_bid = fill_probability(δ_bid, A, k, Δt)

    # Independent Bernoulli draws
    ask_filled = rand(rng) < p_ask
    bid_filled = rand(rng) < p_bid

    return FillOutcome(bid_filled, ask_filled, bid_price, ask_price, V_market)
end

#= 
Computes fill probabilities assuming regime i.

Used by the belief updater to compute 
P(fill_outcome | regime = j) for the Hamilton filter's fill likelihood step.
=#
function fill_probability_for_regime(
    bid_price::Float64,
    ask_price::Float64,
    V_market_j::Float64,
    config::SimConfig
)
    δ_ask = ask_price - V_market_j # TODO: need to make a note explaining what δ represents
    δ_bid = V_market_j - bid_price

    p_ask = fill_probability(δ_ask, config.A, config.k, config.Δt)
    p_bid = fill_probability(δ_bid, config.A, config.k, config.Δt)

    return (; p_bid, p_ask)
end

#=
Probability of observing an exact fill outcome (ex: ask filled, bid not)

The four possible outcomes and their probabilities:
- No fill:        (1 - p_bid) × (1 - p_ask)
- Bid only:       p_bid × (1 - p_ask)
- Ask only:       (1 - p_bid) × p_ask
- Both:           p_bid × p_ask

Used in the Hamilton Filter
=#
function fill_outcome_likelihood(
    fill::FillOutcome,
    bid_price::Float64,
    ask_price::Float64,
    V_market_i::Float64,
    config::SimConfig
)
    probs = fill_probability_for_regime(bid_price, ask_price, V_market_i, config)
    p_bid, p_ask = probs.p_bid, probs.p_ask

    # Joint probability of the exact observed outcome
    p_bid_outcome = fill.bid_filled  ? p_bid  : (1.0 - p_bid)
    p_ask_outcome = fill.ask_filled  ? p_ask  : (1.0 - p_ask)

    return p_bid_outcome * p_ask_outcome  # independent Bernoulli
end