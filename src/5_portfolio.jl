# Compute portfolio-level Greeks and value under a single believed σ.
# Returns hat_Δ_P (net delta including spot), hat_Γ_P, and option value.
function compute_portfolio(
    portfolio::Portfolio,
    options::Vector{OptionContract},
    S::Float64,
    τ::Float64,
    σ::Float64,     # agent's believed vol
    r::Float64,
)
    option_value = 0.0
    Δ_options    = 0.0
    Γ_net        = 0.0

    for (q, opt) in zip(portfolio.option_quantities, options)
        q == 0 && continue
        bs = bs_all(S, opt.K, τ, σ, r; call = opt.is_call)
        option_value += q * bs.price
        Δ_options    += q * bs.Δ
        Γ_net        += q * bs.Γ
    end

    hat_Δ_P = Δ_options + portfolio.q_spot
    return (; option_value, hat_Δ_P, Γ_net)
end

# Compute total portfolio wealth using the TRUE market option value.
# Used for the P&L component of the reward.
function compute_true_wealth(
    portfolio::Portfolio,
    options::Vector{OptionContract},
    S::Float64,
    τ::Float64,
    vs::VolState,
    r::Float64,
)
    market_belief = perfect_regime_belief(vs)
    option_value  = 0.0

    for (q, opt) in zip(portfolio.option_quantities, options)
        q == 0 && continue
        V_market = bs_all_belief_weighted(
            S, opt.K, τ, vs.vm.σ_levels, market_belief, r; call = opt.is_call
        ).price
        option_value += q * V_market
    end

    return option_value + portfolio.q_spot * S + portfolio.cash
end

# Hedge trade: adjust spot position so net portfolio delta equals Δ_target.
# Δ_target is the agent's chosen residual exposure (continuous, from action).
function execute_hedge!(
    portfolio::Portfolio,
    Δ_target::Float64,
    current_hat_Δ_P::Float64,
    S::Float64,
    config::SimConfig,
)
    shares_to_trade = Δ_target - current_hat_Δ_P
    hedge_cost      = config.κ * abs(shares_to_trade) * S

    portfolio.cash   -= shares_to_trade * S + hedge_cost
    portfolio.q_spot += shares_to_trade

    return shares_to_trade, hedge_cost
end

# Reward = true P&L (wealth change using market V*) − risk penalty on chosen exposure.
function compute_reward(
    wealth_before::Float64,
    wealth_after::Float64,
    Δ_target::Float64,
    config::SimConfig,
)
    return (wealth_after - wealth_before) - config.φ * Δ_target^2
end

# Build the agent's observable state from current portfolio + market data + particle filter output.
function build_agent_state(
    portfolio::Portfolio,
    options::Vector{OptionContract},
    S::Float64,
    τ::Float64,
    σ_hat::Float64,
    r::Float64,
    r_t::Float64,   # log return this step (NaN at episode start)
    f_t::Int,
)
    if isempty(options) || τ <= 0.0
        return AgentState(r_t, 0.0, 0.0, f_t, τ, σ_hat, 0.0, 0.0, S)
    end

    opt = options[1]
    q   = portfolio.option_quantities[1]
    bs  = bs_all(S, opt.K, τ, σ_hat, r; call = opt.is_call)

    hat_Δ_P = q * bs.Δ + portfolio.q_spot
    hat_Γ_P = q * bs.Γ

    return AgentState(r_t, hat_Δ_P, hat_Γ_P, f_t, τ, σ_hat, bs.price, bs.Δ, S)
end

# Settle expired options at intrinsic value and begin a new ATM contract.
# only the option book resets.
function reset_for_new_option!(
    portfolio::Portfolio,
    options::Vector{OptionContract},
    S_new::Float64,
)
    for (q, opt) in zip(portfolio.option_quantities, options)
        q == 0 && continue
        intrinsic = opt.is_call ? max(S_new - opt.K, 0.0) : max(opt.K - S_new, 0.0)
        portfolio.cash += q * intrinsic
    end

    empty!(portfolio.option_quantities)
    empty!(options)

    push!(options, OptionContract(round(S_new), true))
    push!(portfolio.option_quantities, 0)
end
