
function compute_portfolio(
    portfolio::Portfolio,
    options::Vector{OptionContract},
    S::Float64,
    τ::Float64,
    σ_regimes::Vector{Float64},
    regime_beliefs::Vector{Float64},
    r::Float64,
)
    option_value = 0.0
    Δ_options = 0.0
    Γ_net     = 0.0
    ν_net     = 0.0
    Θ_net     = 0.0

    for (q, opt) in zip(portfolio.option_quantities, options)
        q == 0 && continue  # skip flat positions to avoid unnecessary BS calls

        bs = bs_all_belief_weighted(S, opt.K, τ, σ_regimes, regime_beliefs, r; call=opt.is_call)

        option_value += q * bs.price
        Δ_options    += q * bs.Δ
        Γ_net        += q * bs.Γ
        ν_net        += q * bs.ν
        Θ_net        += q * bs.Θ
    end

    portfolio_value = option_value + (portfolio.q_spot * S)

    return (; portfolio_value, Δ_options, Γ_net, ν_net, Θ_net)
end

function update_from_fills!(
    portfolio::Portfolio,
    fill::FillOutcome,
    option_idx::Int # which option is this for. Only 1 option in this proj for now
)
    if fill.bid_filled
        portfolio.option_quantities[option_idx] += 1
        portfolio.cash -= fill.bid_price
    end

    if fill.ask_filled
        portfolio.option_quantities[option_idx] -= 1
        portfolio.cash += fill.ask_price
    end
end

function execute_hedge!(
    portfolio::Portfolio,
    target_Δ::Float64,
    Δ_options::Float64,
    S::Float64,
    config::SimConfig
)
    shares_to_trade = target_Δ - Δ_options
    hedge_cost      = config.κ * abs(shares_to_trade) * S

    portfolio.cash   -= shares_to_trade * S + hedge_cost
    portfolio.q_spot  = portfolio.q_spot + shares_to_trade

    return shares_to_trade, hedge_cost
end

function compute_reward(
    wealth_before::Float64,
    wealth_after::Float64,
    net_Δ::Float64,
    config::SimConfig
)
    pnl          = wealth_after - wealth_before
    risk_penalty = config.φ * net_Δ^2
    return pnl - risk_penalty
end

# Construct the agent-observable AgentState from current portfolio + market data.
function build_agent_state(
    portfolio::Portfolio,
    options::Vector{OptionContract},
    S::Float64,
    τ::Float64,
    regime_belief::Vector{Float64},
    σ_regimes::Vector{Float64},
    r::Float64,
)
    port = compute_portfolio(portfolio, options, S, τ, σ_regimes, regime_belief, r)
    net_Δ = port.Δ_options + portfolio.q_spot

    return AgentState(
        S,
        τ,
        net_Δ,
        port.Γ_net,
        port.ν_net,
        port.Θ_net,
        regime_belief
    )
end

# Call this when current options expire to settle
# S carries over and determines the new K
function reset_for_new_option!(
    portfolio::Portfolio,
    options::Vector{OptionContract},
    S::Float64
)
    # Settle all current positions at intrinsic value
    for (q, opt) in zip(portfolio.option_quantities, options)
        q == 0 && continue
        intrinsic = opt.is_call ? max(S - opt.K, 0.0) : max(opt.K - S, 0.0)
        portfolio.cash += q * intrinsic
    end

    # Clear old option book
    empty!(portfolio.option_quantities)
    empty!(options)

    # Start new ATM call
    new_K = round(S)
    push!(options, OptionContract(new_K, true))
    push!(portfolio.option_quantities, 0)
end

