# ============================================================
# Section 1: Dependencies & Setup
# ============================================================
include("9_mcts_mdp.jl") #

using Plots
using Plots.PlotMeasures
using StatsPlots
using Printf
using Statistics: mean, std, median, quantile
using Random: MersenneTwister

# ============================================================
# Section 2: Oracle σ helper
# ============================================================
function oracle_σ(env::EnvironmentState)::Float64
    vs      = env.vol_state
    weights = perfect_regime_belief(vs)
    return sqrt(sum(weights .* vs.vm.σ_levels .^ 2))
end

# ============================================================
# Section 3: Policy registry
# ============================================================
const POLICIES = [
    (fn = glft_ww_policy,    name = "GLF-T + WW",    color = :steelblue,      ls = :solid),
    (fn = glft_naive_policy, name = "GLF-T + Naive",  color = :darkorange,     ls = :dash),
    (fn = naive_ww_policy,   name = "Naive + WW",    color = :mediumseagreen, ls = :dot),
    (fn = naive_naive_policy,name = "Naive + Naive",  color = :crimson,        ls = :dashdot),
]

# ============================================================
# Section 4: Episode runner
# ============================================================
struct EpisodeData
    total_reward::Float64
    steps::Vector{StepInfo}
    spread_values::Vector{Float64}
    hedge_traded::Vector{Bool}
    hat_Δ_P_pre::Vector{Float64}
    regime_history::Vector{Int}
    τ_history::Vector{Float64}
    cumulative_reward::Vector{Float64}
end

function run_episode(
    policy_fn, vol_model::VolModel, config::SimConfig, rng::AbstractRNG;
    n_particles::Int = 500, use_oracle::Bool = false
)::EpisodeData
    pf = ParticleFilter(n_particles)
    env = EnvironmentState(
        AgentState(NaN, 0.0, 0.0, 0, config.T_option * config.Δt,
                   get_σ_hat(pf), 0.0, 0.0, config.S0),
        VolState(vol_model),
        OptionContract[OptionContract(round(config.S0), true)],
        0
    )
    portfolio = Portfolio()
    push!(portfolio.option_quantities, 0)
    initialize_episode!(env, portfolio, vol_model, config, pf)

    if use_oracle
        ws_init = perfect_regime_belief(env.vol_state)
        bs_init = bs_all_belief_weighted(
            env.agent_state.S, env.current_options[1].K, env.agent_state.τ,
            env.vol_state.vm.σ_levels, ws_init, config.r; call = env.current_options[1].is_call
        )
        env.agent_state = AgentState(
            NaN, portfolio.option_quantities[1] * bs_init.Δ + portfolio.q_spot, 
            portfolio.option_quantities[1] * bs_init.Γ,
            0, env.agent_state.τ, sqrt(sum(ws_init .* env.vol_state.vm.σ_levels .^ 2)),
            bs_init.price, bs_init.Δ, env.agent_state.S
        )
    end

    steps, spreads, traded, deltas, regimes, taus, rewards = [], [], [], [], [], [], []
    total_reward, done = 0.0, false

    while !done
        σ = oracle_σ(env)
        action = policy_fn(env, portfolio, config, σ)
        push!(spreads, action.δ); push!(traded, action.Δ_target != env.agent_state.hat_Δ_P)
        push!(deltas, abs(env.agent_state.hat_Δ_P)); push!(regimes, env.vol_state.regime_idx)
        push!(taus, env.agent_state.τ)

        _, reward, done, info = step_environment!(env, portfolio, pf, action, config, rng;
            oracle_regime = use_oracle ? (env.vol_state.vm.σ_levels, perfect_regime_belief(env.vol_state)) : nothing)

        push!(steps, info); total_reward += reward; push!(rewards, total_reward)
    end
    return EpisodeData(total_reward, steps, spreads, traded, deltas, regimes, taus, rewards)
end

# ============================================================
# Section 5: Monte Carlo Evaluation
# ============================================================
struct PolicyResults
    name::String; episode_rewards::Vector{Float64}; mean_reward::Float64; std_reward::Float64
    sharpe::Float64; mean_spread::Float64; hedge_freq::Float64; mean_abs_hat_Δ_P::Float64
    mean_hedge_cost_per_episode::Float64; τ_buckets::Vector{Float64}; mean_spread_by_τ::Vector{Float64}
end

function evaluate_policy(policy_fn, name, vol_model, config, n_episodes, rng; n_particles=500, use_oracle=false)::PolicyResults
    rewards, costs, spreads, traded, deltas, taus = [], [], [], [], [], []
    for _ in 1:n_episodes
        ep = run_episode(policy_fn, vol_model, config, rng; n_particles, use_oracle)
        push!(rewards, ep.total_reward); push!(costs, sum(s.hedge_cost for s in ep.steps))
        append!(spreads, ep.spread_values); append!(traded, ep.hedge_traded)
        append!(deltas, ep.hat_Δ_P_pre); append!(taus, ep.τ_history)
    end
    
    τ_days = Int.(round.(taus .* 252)); τ_unique = sort(unique(τ_days))
    spread_by_τ = [mean(spreads[τ_days .== t]) for t in τ_unique]

    return PolicyResults(name, rewards, mean(rewards), std(rewards), mean(rewards)/std(rewards), 
                         mean(spreads), mean(traded), mean(deltas), mean(costs), Float64.(τ_unique), spread_by_τ)
end

# ============================================================
# Section 6: Refactored Figures
# ============================================================

function fig_pnl_distributions(results_const, results_hardy; save_path = "results/fig1_pnl_distributions.png")
    # Legend moved to lower left to avoid obstructing distributions
    p1 = plot(title="Constant Volatility", xlabel="P&L", ylabel="Freq", legend=:bottomleft)
    p2 = plot(title="Regime-Switching", xlabel="P&L", legend=:bottomleft)
    for (i, pol) in enumerate(POLICIES)
        histogram!(p1, results_const[i].episode_rewards, bins=40, alpha=0.4, label=pol.name, color=pol.color, normalize=:probability, lw=0)
        histogram!(p2, results_hardy[i].episode_rewards, bins=40, alpha=0.4, label=pol.name, color=pol.color, normalize=:probability, lw=0)
    end
    fig = plot(p1, p2, layout=(1,2), size=(900, 380), margin=5mm, link=:both)
    savefig(fig, save_path); return fig
end

function fig_spread_vs_tau(results_const, results_hardy; save_path = "results/fig2_spread_vs_tau.png")
    p1 = plot(title="Constant Volatility", xlabel="Days", ylabel="Spread", legend=:bottomleft, xflip=true)
    p2 = plot(title="Regime-Switching", xlabel="Days", legend=:bottomleft, xflip=true)
    for (i, pol) in enumerate(POLICIES)
        plot!(p1, results_const[i].τ_buckets, results_const[i].mean_spread_by_τ, label=pol.name, color=pol.color, ls=pol.ls, lw=2)
        plot!(p2, results_hardy[i].τ_buckets, results_hardy[i].mean_spread_by_τ, label=pol.name, color=pol.color, ls=pol.ls, lw=2)
    end
    fig = plot(p1, p2, layout=(1,2), size=(900, 380), margin=5mm)
    savefig(fig, save_path); return fig
end

function fig_cumulative_pnl(vol_model, config; env_name, seed=1234, save_path, n_particles=500, n_chain=5)
    # n_chain=5 chains five episodes to show multiple regime transitions
    p = plot(title="Cumulative P&L — $env_name ($n_chain Episodes)", 
             xlabel="Timestep", ylabel="P&L", legend=:bottomleft)
    
    regime_master = Int[]
    for (i, pol) in enumerate(POLICIES)
        combined_rewards, curr_total = [0.0], 0.0
        local_rng = MersenneTwister(seed + i)
        
        for _ in 1:n_chain
            ep = run_episode(pol.fn, vol_model, config, local_rng; n_particles)
            for r in ep.cumulative_reward push!(combined_rewards, r + curr_total) end
            curr_total = combined_rewards[end]
            if i == 1 append!(regime_master, ep.regime_history) end
        end
        plot!(p, combined_rewards, label=pol.name, color=pol.color, ls=pol.ls, lw=1.5)
    end
    
    # Red shading appears as distinct bands across multiple regime transitions
    if length(vol_model.σ_levels) > 1
        _shade_regimes!(p, regime_master)
    end
    hline!(p, [0.0], color=:black, ls=:dash, alpha=0.5, label="")
    mkpath(dirname(save_path)); savefig(p, save_path); return p
end

# ============================================================
# Section 7: MCTS Analysis & Comparison Figures
# ============================================================

struct PolicyTrace
    δ_history::Vector{Float64}
    Δ_target_history::Vector{Float64}
    regime_history::Vector{Int}
    τ_history::Vector{Float64}
    S_history::Vector{Float64}
    step_rewards::Vector{Float64}
    cumulative_reward::Vector{Float64}
end

function run_policy_trace(policy_fn, vm::VolModel, config::SimConfig, seed::Int;
                          n_particles::Int = 500, use_oracle::Bool = true,
                          regime_init::Union{Nothing,Int} = nothing)::PolicyTrace
    rng = MersenneTwister(seed)
    pf  = ParticleFilter(n_particles)
    vs  = isnothing(regime_init) ? VolState(vm) : VolState(vm, regime_init)
    env = EnvironmentState(
        AgentState(NaN, 0.0, 0.0, 0, config.T_option * config.Δt,
                   get_σ_hat(pf), 0.0, 0.0, config.S0),
        vs,
        OptionContract[OptionContract(round(config.S0), true)],
        0
    )
    portfolio = Portfolio()
    push!(portfolio.option_quantities, 0)
    initialize_episode!(env, portfolio, vm, config, pf)

    if use_oracle
        ws_init = perfect_regime_belief(env.vol_state)
        bs_init = bs_all_belief_weighted(
            env.agent_state.S, env.current_options[1].K, env.agent_state.τ,
            env.vol_state.vm.σ_levels, ws_init, config.r; call = env.current_options[1].is_call
        )
        q0 = portfolio.option_quantities[1]
        env.agent_state = AgentState(
            NaN, q0 * bs_init.Δ + portfolio.q_spot, q0 * bs_init.Γ,
            0, env.agent_state.τ, sqrt(sum(ws_init .* env.vol_state.vm.σ_levels .^ 2)),
            bs_init.price, bs_init.Δ, env.agent_state.S
        )
    end

    δs, Δts, regs, taus, Ss, rwds, cum = Float64[], Float64[], Int[], Float64[], Float64[], Float64[], Float64[]
    total = 0.0; done = false
    while !done
        act = policy_fn(env, portfolio, config, oracle_σ(env))
        push!(δs, act.δ); push!(Δts, act.Δ_target)
        push!(regs, env.vol_state.regime_idx); push!(taus, env.agent_state.τ); push!(Ss, env.agent_state.S)
        _, reward, done, _ = step_environment!(env, portfolio, pf, act, config, rng;
            oracle_regime = use_oracle ? (env.vol_state.vm.σ_levels, perfect_regime_belief(env.vol_state)) : nothing)
        total += reward; push!(rwds, reward); push!(cum, total)
    end
    return PolicyTrace(δs, Δts, regs, taus, Ss, rwds, cum)
end

function run_mcts_trace(vm::VolModel, config::SimConfig, seed::Int;
                        n_queries::Int = 200, max_depth::Int = 20,
                        regime_init::Union{Nothing,Int} = nothing)::PolicyTrace
    mdp      = OptionsMM_MDP(config, vm)
    solver   = make_mcts_solver(config, vm; n_queries, max_depth, seed)
    rng_env  = MersenneTwister(seed)   # same seed as run_policy_trace → matched env trajectory
    pf_dummy = ParticleFilter(10)

    vs  = isnothing(regime_init) ? VolState(vm) : VolState(vm, regime_init)
    env = EnvironmentState(
        AgentState(NaN, 0.0, 0.0, 0, config.T_option * config.Δt, 0.2, 0.0, 0.0, config.S0),
        vs,
        OptionContract[OptionContract(Float64(round(config.S0)), true)],
        0,
    )
    portfolio = Portfolio()
    push!(portfolio.option_quantities, 0)
    initialize_episode!(env, portfolio, vm, config, pf_dummy)

    mdp_state = MDPState(
        config.S0, env.vol_state.regime_idx, 0,
        Float64(round(config.S0)), 0.0, 0.0, config.T_option * config.Δt, 0,
    )

    δs, Δts, regs, taus, Ss, rwds, cum = Float64[], Float64[], Int[], Float64[], Float64[], Float64[], Float64[]
    total = 0.0; done = false
    while !done
        planner = solve(solver, mdp)
        act     = POMDPs.action(planner, mdp_state)
        push!(δs, act.δ); push!(Δts, act.Δ_target)
        push!(regs, env.vol_state.regime_idx); push!(taus, env.agent_state.τ); push!(Ss, env.agent_state.S)

        ws_cur = vm.transition_matrix[env.vol_state.regime_idx, :]
        _, reward, done, _ = step_environment!(
            env, portfolio, pf_dummy, act, config, rng_env;
            oracle_regime = (vm.σ_levels, ws_cur),
        )
        total += reward; push!(rwds, reward); push!(cum, total)

        mdp_state = MDPState(
            env.agent_state.S, env.vol_state.regime_idx,
            portfolio.option_quantities[1], env.current_options[1].K,
            portfolio.q_spot, portfolio.cash, env.agent_state.τ, env.options_completed,
        )
    end
    return PolicyTrace(δs, Δts, regs, taus, Ss, rwds, cum)
end

function _shade_regimes!(p, regimes)
    in_high, start_t, labeled = false, 0, false
    for (t, r) in enumerate(regimes)
        if r == 2 && !in_high
            start_t = t; in_high = true
        elseif r != 2 && in_high
            vspan!(p, [start_t, t - 1]; color=:crimson, alpha=0.12,
                   label = labeled ? "" : "High-Vol Regime")
            labeled = true; in_high = false
        end
    end
    if in_high
        vspan!(p, [start_t, length(regimes)]; color=:crimson, alpha=0.12,
               label = labeled ? "" : "High-Vol Regime")
    end
end

function fig_mcts_trajectory(mcts_trace::PolicyTrace, glft_trace::PolicyTrace;
                              config::SimConfig = SIM_CONFIG,
                              save_path = "results/fig5_mcts_trajectory.png")
    T  = length(mcts_trace.δ_history)
    ts = 1:T
    expiry_steps = config.T_option : config.T_option : T

    add_expiry!(p) = for s in expiry_steps
        vline!(p, [s]; color=:gray60, ls=:dot, lw=0.8, alpha=0.7, label="")
    end

    # (a) Spot price path with regime shading
    p1 = plot(ts, mcts_trace.S_history; color=:black, lw=1.5, label="Spot",
              title="(a) Spot Price", xlabel="", ylabel="S (\$)", legend=:topright)
    _shade_regimes!(p1, mcts_trace.regime_history); add_expiry!(p1)

    # (b) Half-spread δ: MCTS vs GLF-T+WW
    p2 = plot(ts, glft_trace.δ_history; color=:darkorange, ls=:dash, lw=1.8, label="GLF-T+WW",
              title="(b) Half-Spread \$\\delta\$", xlabel="", ylabel="\$\\delta\$ (\$)", legend=:topright)
    plot!(p2, ts, mcts_trace.δ_history; color=:steelblue, lw=1.5, label="MCTS-DPW")
    _shade_regimes!(p2, mcts_trace.regime_history); add_expiry!(p2)

    # (c) Hedge target Δ_target: MCTS vs GLF-T+WW
    p3 = plot(ts, glft_trace.Δ_target_history; color=:darkorange, ls=:dash, lw=1.8, label="GLF-T+WW",
              title="(c) Hedge Target \$\\Delta_{\\rm target}\$", xlabel="Timestep",
              ylabel="\$\\Delta_{\\rm target}\$", legend=:bottomleft)
    plot!(p3, ts, mcts_trace.Δ_target_history; color=:steelblue, lw=1.5, label="MCTS-DPW")
    hline!(p3, [0.0]; color=:black, ls=:dot, alpha=0.4, label="")
    _shade_regimes!(p3, mcts_trace.regime_history); add_expiry!(p3)

    # (d) Cumulative P&L on the shared environment trajectory
    p4 = plot([0; collect(ts)], [0.0; glft_trace.cumulative_reward];
              color=:darkorange, ls=:dash, lw=2, label="GLF-T+WW",
              title="(d) Cumulative P&L", xlabel="Timestep", ylabel="P&L (\$)", legend=:topleft)
    plot!(p4, [0; collect(ts)], [0.0; mcts_trace.cumulative_reward];
          color=:steelblue, lw=2, label="MCTS-DPW")
    hline!(p4, [0.0]; color=:black, ls=:dot, alpha=0.4, label="")
    _shade_regimes!(p4, mcts_trace.regime_history); add_expiry!(p4)

    fig = plot(p1, p2, p3, p4; layout=(2, 2), size=(1100, 750), margin=6mm)
    mkpath(dirname(save_path)); savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

function fig_mcts_pnl_comparison(mcts_rewards::Vector{Float64}, glft_rewards::Vector{Float64};
                                   save_path = "results/fig6_mcts_pnl_comparison.png")
    μm, σm = mean(mcts_rewards), std(mcts_rewards)
    μg, σg = mean(glft_rewards), std(glft_rewards)
    n       = length(mcts_rewards)

    p = plot(; title = "P&L Distribution: MCTS-DPW vs GLF-T+WW  (Hardy, $n Episodes)",
               xlabel = "Episode P&L (\$)", ylabel = "Relative Frequency", legend = :topleft)
    histogram!(p, glft_rewards; bins=50, alpha=0.55, normalize=:probability, lw=0, color=:darkorange,
               label = "GLF-T+WW  μ=\$$(round(μg,digits=2))  Sharpe=$(round(μg/σg,digits=2))")
    histogram!(p, mcts_rewards; bins=50, alpha=0.55, normalize=:probability, lw=0, color=:steelblue,
               label = "MCTS-DPW  μ=\$$(round(μm,digits=2))  Sharpe=$(round(μm/σm,digits=2))")
    vline!(p, [μg]; color=:darkorange, lw=2, ls=:dash, label="")
    vline!(p, [μm]; color=:steelblue,  lw=2, ls=:dash, label="")

    mkpath(dirname(save_path)); savefig(p, save_path)
    println("Saved: $save_path")
    return p
end

function fig_mcts_action_scatter(mcts_trace::PolicyTrace, glft_trace::PolicyTrace;
                                  save_path = "results/fig7_mcts_action_scatter.png")
    τm = mcts_trace.τ_history .* 252
    τg = glft_trace.τ_history .* 252
    Δm = abs.(mcts_trace.Δ_target_history)
    Δg = abs.(glft_trace.Δ_target_history)

    # (a) Spread vs time-to-expiry: shows how each policy adjusts δ as τ decays
    p1 = plot(; title="(a) Spread vs Time-to-Expiry", xlabel="Days to Expiry",
                ylabel="Half-Spread \$\\delta\$ (\$)", legend=:topleft, xflip=true)
    scatter!(p1, τg, glft_trace.δ_history; color=:darkorange, alpha=0.6, ms=4,
             markerstrokewidth=0, label="GLF-T+WW")
    scatter!(p1, τm, mcts_trace.δ_history; color=:steelblue, alpha=0.6, ms=4,
             markerstrokewidth=0, label="MCTS-DPW")

    # (b) Spread vs |Δ_target|: shows inventory-risk sensitivity of each policy's quoting
    p2 = plot(; title="(b) Spread vs |Hedge Target|", xlabel="\$|\\Delta_{\\rm target}|\$",
                ylabel="Half-Spread \$\\delta\$ (\$)", legend=:topleft)
    scatter!(p2, Δg, glft_trace.δ_history; color=:darkorange, alpha=0.6, ms=4,
             markerstrokewidth=0, label="GLF-T+WW")
    scatter!(p2, Δm, mcts_trace.δ_history; color=:steelblue, alpha=0.6, ms=4,
             markerstrokewidth=0, label="MCTS-DPW")

    fig = plot(p1, p2; layout=(1, 2), size=(950, 420), margin=6mm)
    mkpath(dirname(save_path)); savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

# ============================================================
# Section 8: Summary & Entry Point
# ============================================================
function print_summary_table(results_const, results_hardy; save_path = "results/table1_summary.txt")

    mkpath(dirname(save_path))
    lines = String[]

    header  = @sprintf("%-20s  %10s  %10s  %10s  %10s  %10s  %10s  %12s",
                        "Policy", "Mean P&L", "Std P&L", "Sharpe",
                        "Spread(\$)", "Hedge%", "|hat_Δ_P|", "Hedge Cost")
    divider = "─"^96

    function block(results, label)
        push!(lines, ""); push!(lines, "Environment: $label")
        push!(lines, divider); push!(lines, header); push!(lines, divider)
        for r in results
            push!(lines, @sprintf(
                "%-20s  %10.4f  %10.4f  %10.4f  %10.4f  %9.1f%%  %10.4f  %12.4f",
                r.name, r.mean_reward, r.std_reward, r.sharpe,
                r.mean_spread, r.hedge_freq * 100, r.mean_abs_hat_Δ_P,
                r.mean_hedge_cost_per_episode))
        end
        push!(lines, divider)
    end

    block(results_const, "Constant Volatility (σ = 0.20)")
    block(results_hardy, "Regime-Switching (Hardy 2001: σ₁=0.121, σ₂=0.269)")

    output = join(lines, "\n")
    println(output)
    open(save_path, "w") do f; println(f, output); end
end

function run_evaluation(; n_episodes=1000, seed=42, n_particles=500)
    rng = MersenneTwister(seed)
    results_const, results_hardy = [], []

    for pol in POLICIES
        push!(results_const, evaluate_policy(pol.fn, pol.name, VM_CONST, SIM_CONFIG, n_episodes, rng; n_particles, use_oracle=true))
        push!(results_hardy, evaluate_policy(pol.fn, pol.name, VM_HARDY, SIM_CONFIG, n_episodes, rng; n_particles, use_oracle=true))
    end

    fig_pnl_distributions(results_const, results_hardy)
    fig_spread_vs_tau(results_const, results_hardy)
    fig_cumulative_pnl(VM_CONST, SIM_CONFIG; env_name="Constant Vol", save_path="results/fig4a_cumulative_pnl_const.png")
    fig_cumulative_pnl(VM_HARDY, SIM_CONFIG; env_name="Hardy Regime", save_path="results/fig4b_cumulative_pnl_hardy.png")

    print_summary_table(results_const, results_hardy)

    # MCTS evaluation
    println("\nEvaluating: MCTS MDP (Hardy, oracle)")
    mcts_results = evaluate_mcts_mdp(VM_HARDY, SIM_CONFIG, n_episodes, seed; n_queries=200, max_depth=20)
    @printf("MCTS MDP — mean P&L = %.4f  std = %.4f  Sharpe = %.4f\n",
            mcts_results.mean_reward, mcts_results.std_reward, mcts_results.sharpe)

    # MCTS figures — both traces use the same seed so they share the same env trajectory
    println("\nGenerating MCTS figures (running trace episode)...")
    trace_seed  = seed + 100
    regime_init = sample(MersenneTwister(trace_seed),
                         1:length(VM_HARDY.σ_levels), Weights(VM_HARDY.stationary_dist))
    mcts_trace  = run_mcts_trace(VM_HARDY, SIM_CONFIG, trace_seed; regime_init)
    glft_trace  = run_policy_trace(glft_ww_policy, VM_HARDY, SIM_CONFIG, trace_seed; regime_init)

    fig_mcts_trajectory(mcts_trace, glft_trace)
    fig_mcts_pnl_comparison(mcts_results.episode_rewards, results_hardy[1].episode_rewards)
    fig_mcts_action_scatter(mcts_trace, glft_trace)

    println("\nEvaluation complete. All outputs in results/")
    return results_const, results_hardy, mcts_results
end

run_evaluation()
