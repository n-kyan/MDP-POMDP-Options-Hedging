struct ValueIterationParams
    γ::Float64
    ε::Float64
    n_samples::Int
    max_iter::Int
end

function default_vi_params()
    return ValueIterationParams(0.999, 0.01, 200, 500)
end

function value_iteration(
    env_params::EnvironmentParams,
    scheme::DiscretizationScheme,
    vi_params::ValueIterationParams
)
    n_s = n_states(scheme)
    n_a = length(HEDGE_FRACTIONS)

    println("Building representative states...")
    rep_states, visit_counts = build_representative_states(env_params, scheme)
    reachable = collect(keys(rep_states))

    println("Reachable bins: $(length(reachable)) / $(n_s)")
    println("Starting value iteration: $(n_a) actions, $(vi_params.n_samples) MC samples")

    V      = zeros(n_s)
    policy = ones(Int, n_s)

    for iter in 1:vi_params.max_iter
        rng     = MersenneTwister(42)
        V_new   = copy(V)
        Δ_v     = 0.0

        for key in reachable
            i_Δ, i_Γ, i_t, i_h = key
            s_idx  = state_index(i_Δ, i_Γ, i_t, i_h, scheme)
            state  = rep_states[key]

            best_val    = -Inf
            best_action = 1

            for a in 1:n_a
                q_val = 0.0
                for _ in 1:vi_params.n_samples
                    Z      = randn(rng)
                    S_next = state.S * exp(
                        (env_params.r - 0.5 * env_params.σ^2) * env_params.dt +
                        env_params.σ * sqrt(env_params.dt) * Z
                    )
                    result   = step_environment(state, a, S_next, env_params)
                    next_key = discretize(result.next_state, env_params, scheme)
                    s_next   = state_index(next_key..., scheme)
                    q_val   += result.reward + vi_params.γ * V[s_next]
                end
                q_val /= vi_params.n_samples

                if q_val > best_val
                    best_val    = q_val
                    best_action = a
                end
            end

            V_new[s_idx]  = best_val
            policy[s_idx] = best_action
            Δ_v           = max(Δ_v, abs(V_new[s_idx] - V[s_idx]))
        end

        V = V_new
        println("Iter $(lpad(iter,3)): max ΔV = $(round(Δ_v, digits=6))")

        if Δ_v < vi_params.ε
            println("Converged at iteration $iter")
            break
        end
    end

    return V, policy
end


function sensitivity_grid()
    dt   = 1/252
    T    = 63
    port = Portfolio([OptionPosition(100.0, T, true, 1.0)])
    vi   = default_vi_params()

    rng_eval   = MersenneTwister(999)
    prices_mat, _ = simulate_gbm(100.0, 0.05, 0.2, dt, T; n_paths=500, rng=rng_eval)

    tc_rates = [0.0, 0.0005, 0.001, 0.005, 0.01]
    λ_vals   = [0.0, 0.1, 0.5, 1.0, 2.0]

    n_tc = length(tc_rates)
    n_λ  = length(λ_vals)

    results = Array{NamedTuple, 2}(undef, n_tc, n_λ)

    println("Running $(n_tc * n_λ) VI solves...")

    for (i, tc_rate) in enumerate(tc_rates)
        for (j, λ) in enumerate(λ_vals)
            print("  tc=$(tc_rate), λ=$(λ)... ")
            tc     = TransactionCostModel(tc_rate)
            params = EnvironmentParams(port, 0.05, 0.2, dt, tc, λ)
            scheme = default_scheme(T)

            V, policy = value_iteration(params, scheme, vi)

            vi_policy = (state, env_params) -> begin
                i_Δ, i_Γ, i_t, i_h = discretize(state, env_params, scheme)
                policy[state_index(i_Δ, i_Γ, i_t, i_h, scheme)]
            end

            vi_res = [run_episode(vi_policy,    prices_mat[:, k], params) for k in 1:500]
            bs_res = [run_episode(bs_benchmark, prices_mat[:, k], params) for k in 1:500]

            vi_pnls = [r.cumulative_pnl for r in vi_res]
            bs_pnls = [r.cumulative_pnl for r in bs_res]

            results[i, j] = (
                vi_mean_pnl   = mean(vi_pnls),
                vi_pnl_std    = std(vi_pnls),
                vi_sharpe     = mean(vi_pnls) / std(vi_pnls) * sqrt(252/T),
                vi_hedge_freq = mean(r.hedge_frequency for r in vi_res),
                vi_tc         = mean(r.total_tc        for r in vi_res),
                bs_mean_pnl   = mean(bs_pnls),
                bs_sharpe     = mean(bs_pnls) / std(bs_pnls) * sqrt(252/T),
                bs_hedge_freq = mean(r.hedge_frequency for r in bs_res),
                bs_tc         = mean(r.total_tc        for r in bs_res),
            )
            println("done. VI hedge freq=$(round(results[i,j].vi_hedge_freq, digits=3))")
        end
    end

    println("\nSaving results...")
    open("results/sensitivity_grid.csv", "w") do f
        println(f, "tc_rate,lambda,vi_mean_pnl,vi_pnl_std,vi_sharpe,vi_hedge_freq,vi_tc,bs_mean_pnl,bs_sharpe,bs_hedge_freq,bs_tc")
        for (i, tc_rate) in enumerate(tc_rates)
            for (j, λ) in enumerate(λ_vals)
                r = results[i, j]
                println(f, "$(tc_rate),$(λ),$(r.vi_mean_pnl),$(r.vi_pnl_std),$(r.vi_sharpe),$(r.vi_hedge_freq),$(r.vi_tc),$(r.bs_mean_pnl),$(r.bs_sharpe),$(r.bs_hedge_freq),$(r.bs_tc)")
            end
        end
    end
    println("Saved to results/sensitivity_grid.csv")

    return results, tc_rates, λ_vals
end

function visualize_policy(policy, scheme, env_params)
    dt = env_params.dt
    T  = env_params.port.positions[1].T

    n_Δ = length(scheme.Δ_edges) - 1
    n_t = length(scheme.time_edges) - 1
    n_h = length(scheme.hedge_ratio_edges) - 1

    Δ_mids = [(scheme.Δ_edges[i] + scheme.Δ_edges[i+1]) / 2.0 for i in 1:n_Δ]
    t_mids = [scheme.time_edges[i] for i in 1:n_t]

    i_Γ_mid = 3
    i_h_mid = 3

    action_grid = zeros(Int, n_Δ, n_t)
    for i_t in 1:n_t
        for i_Δ in 1:n_Δ
            s_idx = state_index(i_Δ, i_Γ_mid, i_t, i_h_mid, scheme)
            action_grid[i_Δ, i_t] = policy[s_idx]
        end
    end

    println("\n=== Policy heatmap: action vs (Δ bin, time bin) ===")
    println("Γ bin fixed at $(i_Γ_mid), hedge_ratio bin fixed at $(i_h_mid)")
    println("Actions: 1=0%, 2=25%, 3=50%, 4=75%, 5=100% hedge")
    println()

    print(rpad("Δ \\ t", 8))
    for i_t in 1:n_t
        print(rpad("t=$(t_mids[i_t])", 8))
    end
    println()
    println("-" ^ (8 + 8*n_t))

    for i_Δ in n_Δ:-1:1
        print(rpad("Δ=$(round(Δ_mids[i_Δ], digits=2))", 8))
        for i_t in 1:n_t
            a = action_grid[i_Δ, i_t]
            label = ["0%", "25%", "50%", "75%", "100%"][a]
            print(rpad(label, 8))
        end
        println()
    end

    println("\n=== Hedge frequency by Δ bin (averaged over time and hedge_ratio) ===")
    println(rpad("Δ mid", 10), rpad("mean action", 14), "hedge fraction")
    for i_Δ in 1:n_Δ
        actions = [action_grid[i_Δ, i_t] for i_t in 1:n_t]
        mean_a  = mean(actions)
        println(
            rpad(round(Δ_mids[i_Δ], digits=2), 10),
            rpad(round(mean_a, digits=2),       14),
            round((mean_a - 1) / 4, digits=2)
        )
    end
end

function verify_value_iteration()
    dt     = 1/252
    T      = 63
    port   = Portfolio([OptionPosition(100.0, T, true, 1.0)])
    tc     = TransactionCostModel(0.001)
    params = EnvironmentParams(port, 0.05, 0.2, dt, tc, 0.1)
    scheme = default_scheme(T)
    vi     = default_vi_params()

    V, policy = value_iteration(params, scheme, vi)

    println("\n=== Policy inspection ===")
    println("Action distribution: $([(a, count(==(a), policy)) for a in 1:5])")
    println("Actions: 1=0% hedge, 2=25%, 3=50%, 4=75%, 5=100%")

    vi_policy = (state, env_params) -> begin
        i_Δ, i_Γ, i_t, i_h = discretize(state, env_params, scheme)
        policy[state_index(i_Δ, i_Γ, i_t, i_h, scheme)]
    end

    println("\n=== Comparison on 500 held-out paths ===")
    rng        = MersenneTwister(999)
    prices_mat, _ = simulate_gbm(100.0, 0.05, 0.2, dt, T; n_paths=500, rng=rng)

    for (label, pol) in [("BS benchmark", bs_benchmark), ("VI policy", vi_policy)]
    results = [run_episode(pol, prices_mat[:, i], params) for i in 1:500]
    
    ep_pnls = [r.cumulative_pnl for r in results]
    cross_sharpe = mean(ep_pnls) / std(ep_pnls) * sqrt(252/T)

    println("\n--- $label ---")
    println("Mean PnL:       $(round(mean(ep_pnls),              digits=4))")
    println("PnL std:        $(round(std(ep_pnls),               digits=4))")
    println("Cross-ep Sharpe:$(round(cross_sharpe,               digits=4))")
    println("Hedge freq:     $(round(mean(r.hedge_frequency for r in results), digits=4))")
    println("Mean TC:        $(round(mean(r.total_tc        for r in results), digits=4))")
    end

    visualize_policy(policy, scheme, params)
end