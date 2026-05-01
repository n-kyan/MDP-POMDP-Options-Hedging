include("7_benchmarks.jl")

using Plots
using Plots.PlotMeasures
using StatsPlots
using Printf
using Statistics: mean, std, median, quantile
using Random: MersenneTwister

# ============================================================
# Section 1: Vol models and shared config
# ============================================================

const VM_HARDY = VolModel(
    [0.121, 0.269],
    transition_matrix = [0.9982 0.0018;
                         0.0022 0.9978]
)
const VM_CONST = VolModel([0.20])
const SIM_CONFIG = SimConfig()

# ============================================================
# Section 2: Oracle σ helper
# ============================================================

# Transition-row-weighted variance-equivalent σ — same information the market uses.
# Benchmark policies receive this so any performance gap vs RL reflects policy quality,
# not information asymmetry.
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
    (fn = naive_ww_policy,   name = "Naive + WW",     color = :mediumseagreen, ls = :dot),
    (fn = naive_naive_policy,name = "Naive + Naive",  color = :crimson,        ls = :dashdot),
]

# ============================================================
# Section 4: Episode runner
# ============================================================

struct EpisodeData
    total_reward::Float64
    steps::Vector{StepInfo}
    spread_values::Vector{Float64}    # δ chosen at each step
    hedge_traded::Vector{Bool}        # true if Δ_target ≠ hat_Δ_P (hedge moved)
    hat_Δ_P_pre::Vector{Float64}      # |hat_Δ_P| before each step
    regime_history::Vector{Int}
    τ_history::Vector{Float64}
    cumulative_reward::Vector{Float64}
end

function run_episode(
    policy_fn,
    vol_model::VolModel,
    config::SimConfig,
    rng::AbstractRNG;
    n_particles::Int = 500,
    use_oracle::Bool = false,
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

    # Oracle benchmarks: rebuild initial agent state with transition-weighted σ,
    # bypassing the particle filter prior (which starts at σ̂ ≈ 0.20).
    if use_oracle
        σ_init = oracle_σ(env)
        env.agent_state = build_agent_state(
            portfolio, env.current_options,
            env.agent_state.S, env.agent_state.τ,
            σ_init, config.r, NaN, 0,
        )
    end

    steps             = StepInfo[]
    spread_values     = Float64[]
    hedge_traded      = Bool[]
    hat_Δ_P_pre       = Float64[]
    regime_history    = Int[]
    τ_history         = Float64[]
    cumulative_reward = Float64[]
    total_reward      = 0.0
    done              = false

    while !done
        σ      = oracle_σ(env)
        action = policy_fn(env, portfolio, config, σ)

        push!(spread_values,  action.δ)
        push!(hedge_traded,   action.Δ_target != env.agent_state.hat_Δ_P)
        push!(hat_Δ_P_pre,    abs(env.agent_state.hat_Δ_P))
        push!(regime_history, env.vol_state.regime_idx)
        push!(τ_history,      env.agent_state.τ)

        _, reward, done, step_info = step_environment!(env, portfolio, pf, action, config, rng;
                                                       σ_hat_override = use_oracle ? σ : NaN)

        push!(steps, step_info)
        total_reward += reward
        push!(cumulative_reward, total_reward)
    end

    return EpisodeData(
        total_reward, steps, spread_values, hedge_traded,
        hat_Δ_P_pre, regime_history, τ_history, cumulative_reward
    )
end

# ============================================================
# Section 5: Monte Carlo evaluation
# ============================================================

struct PolicyResults
    name::String
    episode_rewards::Vector{Float64}
    mean_reward::Float64
    std_reward::Float64
    sharpe::Float64
    mean_spread::Float64
    hedge_freq::Float64
    mean_abs_hat_Δ_P::Float64
    mean_hedge_cost_per_episode::Float64
    τ_buckets::Vector{Float64}
    mean_spread_by_τ::Vector{Float64}
end

function evaluate_policy(
    policy_fn,
    name::String,
    vol_model::VolModel,
    config::SimConfig,
    n_episodes::Int,
    rng::AbstractRNG;
    n_particles::Int = 500,
    use_oracle::Bool = false,
)::PolicyResults
    episode_rewards     = Float64[]
    episode_hedge_costs = Float64[]
    all_spread          = Float64[]
    all_hedge_traded    = Bool[]
    all_abs_hat_Δ_P     = Float64[]
    all_τ               = Float64[]
    all_spread_for_τ    = Float64[]

    for _ in 1:n_episodes
        ep = run_episode(policy_fn, vol_model, config, rng; n_particles, use_oracle)
        push!(episode_rewards,     ep.total_reward)
        push!(episode_hedge_costs, sum(s.hedge_cost for s in ep.steps))
        append!(all_spread,        ep.spread_values)
        append!(all_hedge_traded,  ep.hedge_traded)
        append!(all_abs_hat_Δ_P,   ep.hat_Δ_P_pre)
        append!(all_τ,             ep.τ_history)
        append!(all_spread_for_τ,  ep.spread_values)
    end

    μ = mean(episode_rewards)
    σ = std(episode_rewards)
    sharpe = σ > 1e-10 ? μ / σ : 0.0

    τ_in_days        = round.(all_τ .* 252)
    τ_unique         = sort(unique(τ_in_days))
    mean_spread_by_τ = [mean(all_spread_for_τ[τ_in_days .== t]) for t in τ_unique]

    return PolicyResults(
        name, episode_rewards, μ, σ, sharpe,
        mean(all_spread),
        mean(all_hedge_traded),
        mean(all_abs_hat_Δ_P),
        mean(episode_hedge_costs),
        Float64.(τ_unique),
        mean_spread_by_τ,
    )
end

# ============================================================
# Section 6: Figures
# ============================================================

function fig_pnl_distributions(
    results_const::Vector{PolicyResults},
    results_hardy::Vector{PolicyResults};
    save_path::String = "results/fig1_pnl_distributions.png"
)
    p1 = plot(title = "Constant Volatility (σ = 0.20)",
              xlabel = "Episode P&L", ylabel = "Frequency",
              legend = :topright, titlefontsize = 10)
    p2 = plot(title = "Regime-Switching (Hardy 2001)",
              xlabel = "Episode P&L", ylabel = "Frequency",
              legend = :topright, titlefontsize = 10)

    for (i, pol) in enumerate(POLICIES)
        histogram!(p1, results_const[i].episode_rewards;
                   bins = 40, alpha = 0.4, label = pol.name,
                   color = pol.color, normalize = :probability)
        histogram!(p2, results_hardy[i].episode_rewards;
                   bins = 40, alpha = 0.4, label = pol.name,
                   color = pol.color, normalize = :probability)
    end

    fig = plot(p1, p2, layout = (1, 2), size = (900, 380), margin = 5mm, dpi = 150)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

function fig_spread_vs_tau(
    results_const::Vector{PolicyResults},
    results_hardy::Vector{PolicyResults};
    save_path::String = "results/fig2_spread_vs_tau.png"
)
    p1 = plot(title = "Constant Volatility",
              xlabel = "Days to Expiry", ylabel = "Mean Half-Spread (\$)",
              legend = :topright, titlefontsize = 10, xflip = true)
    p2 = plot(title = "Regime-Switching",
              xlabel = "Days to Expiry", ylabel = "Mean Half-Spread (\$)",
              legend = :topright, titlefontsize = 10, xflip = true)

    for (i, pol) in enumerate(POLICIES)
        plot!(p1, results_const[i].τ_buckets, results_const[i].mean_spread_by_τ;
              label = pol.name, color = pol.color, linestyle = pol.ls, linewidth = 2)
        plot!(p2, results_hardy[i].τ_buckets, results_hardy[i].mean_spread_by_τ;
              label = pol.name, color = pol.color, linestyle = pol.ls, linewidth = 2)
    end

    fig = plot(p1, p2, layout = (1, 2), size = (900, 380), margin = 5mm, dpi = 150)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

function fig_hedge_behavior(
    results_const::Vector{PolicyResults},
    results_hardy::Vector{PolicyResults};
    save_path::String = "results/fig3_hedge_behavior.png"
)
    names       = [pol.name for pol in POLICIES]
    hedge_const = [r.hedge_freq        for r in results_const]
    hedge_hardy = [r.hedge_freq        for r in results_hardy]
    Δ_const     = [r.mean_abs_hat_Δ_P  for r in results_const]
    Δ_hardy     = [r.mean_abs_hat_Δ_P  for r in results_hardy]

    p1 = groupedbar(hcat(hedge_const, hedge_hardy);
                    bar_position = :dodge, bar_width = 0.6,
                    xticks = (1:4, names), xrotation = 15,
                    ylabel = "Hedge Trade Frequency", title = "Hedge Frequency",
                    label = ["Const Vol" "Hardy"],
                    color = [:steelblue :darkorange], titlefontsize = 10)
    p2 = groupedbar(hcat(Δ_const, Δ_hardy);
                    bar_position = :dodge, bar_width = 0.6,
                    xticks = (1:4, names), xrotation = 15,
                    ylabel = "Mean |hat_Δ_P|", title = "Mean Absolute Net Delta",
                    label = ["Const Vol" "Hardy"],
                    color = [:steelblue :darkorange], titlefontsize = 10)

    fig = plot(p1, p2, layout = (1, 2), size = (900, 400), margin = 8mm, dpi = 150)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

function fig_cumulative_pnl(
    vol_model::VolModel,
    config::SimConfig;
    env_name::String,
    seed::Int = 1234,
    save_path::String = "results/fig4_cumulative_pnl.png",
    n_particles::Int = 500,
)
    p = plot(title = "Cumulative P&L — $env_name (single episode)",
             xlabel = "Timestep", ylabel = "Cumulative P&L",
             legend = :topleft, titlefontsize = 10)

    if length(vol_model.σ_levels) > 1
        ep_ref = run_episode(POLICIES[1].fn, vol_model, config,
                             MersenneTwister(seed); n_particles)
        in_high_vol  = false
        region_start = 0
        shaded_first = false
        for (t, reg) in enumerate(ep_ref.regime_history)
            if reg == 2 && !in_high_vol
                region_start = t; in_high_vol = true
            elseif reg != 2 && in_high_vol
                lbl = shaded_first ? nothing : "High-Vol Regime"
                vspan!(p, region_start, t - 1; alpha = 0.12, color = :crimson, label = lbl)
                shaded_first = true; in_high_vol = false
            end
        end
        if in_high_vol
            lbl = shaded_first ? nothing : "High-Vol Regime"
            vspan!(p, region_start, length(ep_ref.regime_history);
                   alpha = 0.12, color = :crimson, label = lbl)
        end
    end

    for pol in POLICIES
        ep = run_episode(pol.fn, vol_model, config, MersenneTwister(seed); n_particles)
        plot!(p, ep.cumulative_reward;
              label = pol.name, color = pol.color, linestyle = pol.ls, linewidth = 1.5)
    end
    hline!(p, [0.0]; color = :black, linestyle = :dash, linewidth = 0.8, label = "")

    fig = plot(p, size = (800, 380), margin = 5mm, dpi = 150)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

# ============================================================
# Section 7: Summary table
# ============================================================

function print_summary_table(
    results_const::Vector{PolicyResults},
    results_hardy::Vector{PolicyResults};
    save_path::String = "results/table1_summary.txt"
)
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
    println("\nSaved: $save_path")
end

# ============================================================
# Section 8: Entry point
# ============================================================

function run_evaluation(; n_episodes::Int = 1_000, seed::Int = 42, n_particles::Int = 500)
    rng = MersenneTwister(seed)

    println("="^60)
    println("Options Market Making — Benchmark Evaluation")
    println("n_episodes=$n_episodes  seed=$seed  n_particles=$n_particles")
    println("="^60)

    results_const = PolicyResults[]
    results_hardy = PolicyResults[]

    for pol in POLICIES
        println("\nEvaluating: $(pol.name)")

        print("  Constant vol... ")
        r_const = evaluate_policy(pol.fn, pol.name, VM_CONST, SIM_CONFIG,
                                  n_episodes, rng; n_particles, use_oracle = true)
        push!(results_const, r_const)
        @printf("done — mean P&L = %.4f, Sharpe = %.4f\n", r_const.mean_reward, r_const.sharpe)

        print("  Hardy regime...  ")
        r_hardy = evaluate_policy(pol.fn, pol.name, VM_HARDY, SIM_CONFIG,
                                  n_episodes, rng; n_particles, use_oracle = true)
        push!(results_hardy, r_hardy)
        @printf("done — mean P&L = %.4f, Sharpe = %.4f\n", r_hardy.mean_reward, r_hardy.sharpe)
    end

    println("\nGenerating figures...")
    fig_pnl_distributions(results_const, results_hardy)
    fig_spread_vs_tau(results_const, results_hardy)
    fig_hedge_behavior(results_const, results_hardy)
    fig_cumulative_pnl(VM_CONST,  SIM_CONFIG; env_name = "Constant Vol",
                       seed = 77, save_path = "results/fig4a_cumulative_pnl_const.png",
                       n_particles)
    fig_cumulative_pnl(VM_HARDY, SIM_CONFIG; env_name = "Hardy Regime-Switching",
                       save_path = "results/fig4b_cumulative_pnl_hardy.png", n_particles)

    println("\nSummary Table:")
    print_summary_table(results_const, results_hardy)

    println("\nEvaluation complete. All outputs in results/")
    return results_const, results_hardy
end
