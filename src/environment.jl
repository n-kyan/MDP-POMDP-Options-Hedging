# environment.jl

# inculdet("types.jl")

struct TransactionCostModel
    rate::Float64
end

struct EnvironmentParams
    port::Portfolio
    r::Float64
    σ::Float64
    dt::Float64
    tc::TransactionCostModel
    λ::Float64
end

struct StepState
    S::Float64
    t::Int
    hedge_position::Float64
end

struct StepResult
    next_state::StepState
    reward::Float64
    options_pnl::Float64
    hedge_pnl::Float64
    tc_cost::Float64
    risk_penalty::Float64
    Δ::Float64
    Γ::Float64
end

function step_environment(
    state::StepState,
    action_idx::Int,
    S_next::Float64,
    params::EnvironmentParams
)
    S, t, hedge = state.S, state.t, state.hedge_position
    port, r, σ, dt, tc, λ = params.port, params.r, params.σ, params.dt, params.tc, params.λ

    Δ, Γ = portfolio_greeks(port, S, r, σ, t, dt)

    f = HEDGE_FRACTIONS[action_idx]
    target_hedge = f * Δ
    trade = target_hedge - hedge
    tc_cost = tc.rate * abs(trade) * S

    V_before = portfolio_price(port, S, r, σ, t, dt)
    V_after = portfolio_price(port, S_next, r, σ, t+1, dt)
    options_pnl = V_after - V_before
    hedge_pnl = -target_hedge * (S_next - S)

    residual_Δ = Δ - target_hedge
    risk_penalty = λ * residual_Δ^2

    reward = options_pnl + hedge_pnl - tc_cost - risk_penalty

    next_state = StepState(S_next, t+1, target_hedge)

    return StepResult(next_state, reward, options_pnl, hedge_pnl, tc_cost, risk_penalty, Δ, Γ)
end

function verify_environment()
    dt   = 1/252
    T    = 63
    port = Portfolio([OptionPosition(100.0, T, true, 1.0)])
    tc   = TransactionCostModel(0.001)
    params = EnvironmentParams(port, 0.05, 0.2, dt, tc, 0.1)

    println("=== Single step ===")
    state  = StepState(100.0, 0, 0.0)
    result = step_environment(state, 3, 101.0, params)
    println("Δ=$(round(result.Δ, digits=4)),  Γ=$(round(result.Γ, digits=4))")
    println("options_pnl=$(round(result.options_pnl, digits=4))")
    println("hedge_pnl=$(round(result.hedge_pnl, digits=4))")
    println("tc_cost=$(round(result.tc_cost, digits=6))")
    println("reward=$(round(result.reward, digits=4))")

    println("\n=== Full episode: BS benchmark (always 100% hedge) ===")
    rng    = MersenneTwister(42)
    prices, _ = simulate_gbm(100.0, 0.05, 0.2, dt, T; n_paths=1, rng=rng)
    state  = StepState(100.0, 0, 0.0)
    total_reward = 0.0
    total_tc     = 0.0
    for i in 1:T
        S_next = prices[i+1, 1]
        result = step_environment(state, 5, S_next, params)
        total_reward += result.reward
        total_tc     += result.tc_cost
        state = result.next_state
    end
    println("Cumulative reward: $(round(total_reward, digits=4))")
    println("Total tc paid:     $(round(total_tc, digits=4))")

    println("\n=== Zero hedge vs full hedge: reward comparison ===")
    rng2   = MersenneTwister(42)
    prices2, _ = simulate_gbm(100.0, 0.05, 0.2, dt, T; n_paths=1, rng=rng2)
    for (label, action) in [("no hedge (0%)", 1), ("full hedge (100%)", 5)]
        state = StepState(100.0, 0, 0.0)
        r_total = 0.0
        for i in 1:T
            result = step_environment(state, action, prices2[i+1, 1], params)
            r_total += result.reward
            state = result.next_state
        end
        println("$(label): cumulative reward = $(round(r_total, digits=4))")
    end
end

struct EpisodeResult
    cumulative_reward::Float64
    cumulative_pnl::Float64
    pnl_variance::Float64
    sharpe::Float64
    hedge_frequency::Float64
    total_tc::Float64
    step_pnls::Vector{Float64}
end

function run_episode(
    policy::Function,
    prices::Vector{Float64},
    params::EnvironmentParams
)
    T = length(prices) - 1
    state = StepState(prices[1], 0, 0.0)

    cumulative_reward = 0.0
    total_tc = 0.0
    step_pnls = Vector{Float64}(undef, T) # review this syntax
    n_trades = 0

    for i in 1:T
        action_idx = policy(state, params)
        S_next = prices[i + 1]
        result = step_environment(state, action_idx, S_next, params)

        step_pnl = result.options_pnl + result.hedge_pnl - result.tc_cost
        step_pnls[i] = step_pnl
        cumulative_reward += result.reward
        total_tc += result.tc_cost

        prev_hedge = state.hedge_position
        if abs(result.next_state.hedge_position - prev_hedge) > 1e-8 # justify this threshold
            n_trades+=1
        end

        state = result.next_state
    end

    cumulative_pnl = sum(step_pnls)
    pnl_variance = var(step_pnls)
    sharpe = mean(step_pnls) / std(step_pnls) * sqrt(252)
    hedge_frequency = n_trades / T
   
    return EpisodeResult(
        cumulative_reward,
        cumulative_pnl,
        pnl_variance,
        sharpe,
        hedge_frequency,
        total_tc,
        step_pnls
    )
end

bs_benchmark = (state, params) -> 5 # what the hell is this
no_hedge = (state, params) -> 1

function verify_episode_runner()
    dt      = 1/252
    T       = 63
    port    = Portfolio([OptionPosition(100.0, T, true, 1.0)])
    n_paths = 500

    rng    = MersenneTwister(42)
    prices_mat, _ = simulate_gbm(100.0, 0.05, 0.2, dt, T; n_paths=n_paths, rng=rng)

    println("=== BS benchmark vs no hedge across $(n_paths) paths ===")
    for (label, policy) in [("BS benchmark", bs_benchmark), ("No hedge", no_hedge)]
        tc     = TransactionCostModel(0.001)
        params = EnvironmentParams(port, 0.05, 0.2, dt, tc, 0.1)

        results = [run_episode(policy, prices_mat[:, i], params) for i in 1:n_paths]

        mean_pnl    = mean(r.cumulative_pnl    for r in results)
        mean_var    = mean(r.pnl_variance       for r in results)
        mean_sharpe = mean(r.sharpe             for r in results)
        mean_hfreq  = mean(r.hedge_frequency    for r in results)
        mean_tc     = mean(r.total_tc           for r in results)

        println("\n--- $(label) ---")
        println("Mean cumulative PnL:  $(round(mean_pnl,    digits=4))")
        println("Mean PnL variance:    $(round(mean_var,    digits=6))")
        println("Mean Sharpe:          $(round(mean_sharpe, digits=4))")
        println("Mean hedge frequency: $(round(mean_hfreq,  digits=4))")
        println("Mean total TC:        $(round(mean_tc,     digits=4))")
    end

    println("\n=== TC sensitivity: BS benchmark, varying transaction cost ===")
    println("tc_rate  | mean PnL  | PnL var   | hedge freq")
    for tc_rate in [0.0, 0.0005, 0.001, 0.005, 0.01]
        tc     = TransactionCostModel(tc_rate)
        params = EnvironmentParams(port, 0.05, 0.2, dt, tc, 0.1)
        results = [run_episode(bs_benchmark, prices_mat[:, i], params) for i in 1:n_paths]
        println(
            "$(rpad(tc_rate, 8)) | ",
            "$(rpad(round(mean(r.cumulative_pnl for r in results), digits=3), 9)) | ",
            "$(rpad(round(mean(r.pnl_variance   for r in results), digits=6), 9)) | ",
            "$(round(mean(r.hedge_frequency for r in results), digits=4))"
        )
    end
end