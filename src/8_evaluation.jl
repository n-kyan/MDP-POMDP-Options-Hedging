# ============================================================
# Module 8: Evaluation and Visualization — 8_evaluation.jl
#
# Runs four benchmark policies (GLF-T+WW, GLF-T+Naive, Naive+WW, Naive+Naive)
# under two volatility environments (constant vol, Hardy regime-switching),
# collects per-step StepInfo data, and produces paper-ready figures.
#
# Depends on: 7_benchmarks.jl (which transitively includes 1–6)
#
# Requires:
#   step_environment! to return (next_state, reward, done, StepInfo)   ← see note below
#   run_benchmark in 7_benchmarks.jl to unpack 4 return values         ← see note below
#
# Usage:
#   julia src/8_evaluation.jl
# ============================================================

include("7_benchmarks.jl")

using Plots
using Plots.PlotMeasures
using StatsPlots
using Printf
using Statistics: mean, std, median, quantile
using Random: MersenneTwister

# ============================================================
# REQUIRED CHANGES IN UPSTREAM MODULES
#
# In 6_environment.jl, change the final return of step_environment! from:
#
#     return next_state, reward, done, log_return, fill
#
# to:
#
#     step_info = StepInfo(log_return, fill, shares_traded, hedge_cost,
#                          wealth_before, wealth_after)
#     return next_state, reward, done, step_info
#
# In 7_benchmarks.jl, inside run_benchmark, change:
#
#     next_state, reward, done, _, _ = step_environment!(...)
#
# to:
#
#     next_state, reward, done, _ = step_environment!(...)
# ============================================================

# ============================================================
# Section 1: Experiment Configuration
# ============================================================

# Hardy (2001) calibrated parameters for S&P 500 (from test_1_2_3.jl)
# σ₁ = 0.121 (low-vol regime),  π₁ ≈ 0.55 (stationary fraction)
# σ₂ = 0.269 (high-vol regime), π₂ ≈ 0.45
# Very high persistence: avg run ~556 days (regime 1), ~455 days (regime 2)
const VM_HARDY = VolModel(
    [0.121, 0.269],
    transition_matrix = [0.9982 0.0018;
                         0.0022 0.9978]
)

# Constant-vol baseline: variance-weighted equivalent of Hardy stationary distribution
# σ_eff = sqrt(π₁σ₁² + π₂σ₂²) = sqrt(0.55×0.121² + 0.45×0.269²) ≈ 0.20
const VM_CONST = VolModel([0.20])

const SIM_CONFIG = SimConfig()

# ============================================================
# Section 2: Oracle σ — gives policies the transition-weighted
# effective vol consistent with the market's belief-weighted pricing
# ============================================================

#=
The market prices at bs_all_belief_weighted(S, K, τ, σ_regimes, perfect_regime_belief(vs), r).
perfect_regime_belief returns the transition row of the current regime, which is the
one-step-ahead distribution over next regimes.

For the analytical benchmark policies (GLF-T, WW) to operate on the same information
as the market, we extract the variance-equivalent scalar σ from that same belief vector:
    σ_eff = sqrt(Σⱼ weights[j] × σⱼ²)

This is the natural single-σ analog of a belief-weighted two-regime world.
For the constant-vol model (single regime), this degenerates to σ₁ = 0.20.
=#
function oracle_σ(env::EnvironmentState)::Float64
    vs = env.vol_state
    weights = perfect_regime_belief(vs)   # transition row = market's one-step belief
    return sqrt(sum(weights .* vs.vm.σ_levels .^ 2))
end

# ============================================================
# Section 3: Policy Closures
# All four policies take (env, portfolio, config, σ) and return MarketMakingAction
# ============================================================

# Policy 1 — GLF-T spread + Whalley-Wilmott hedge
# The sophisticated policy: both components are analytically optimal
function policy_glft_ww(env, portfolio, config, σ)
    return glft_ww_policy(env, portfolio, config, σ)
end

# Policy 2 — GLF-T spread + Naive hedge (always target net_Δ = 0)
# Isolates the contribution of WW hedging vs naive:
# any performance difference vs policy_glft_ww is attributable purely to hedging
function policy_glft_naive(env, portfolio, config, σ)
    return MarketMakingAction(
        glft_spread_idx(env, config, σ),
        naive_hedge_idx(config)
    )
end

# Policy 3 — Naive spread (fixed) + Whalley-Wilmott hedge
# Isolates the contribution of GLF-T spread-setting vs naive:
# any performance difference vs policy_naive_naive is attributable purely to spread
function policy_naive_ww(env, portfolio, config, σ)
    return MarketMakingAction(
        symmetric_spread_idx(2),   # index 2 = 0.05 half-spread (tightest fixed level)
        ww_hedge_idx(env, config, σ)
    )
end

# Policy 4 — Naive spread + Naive hedge
# Pure baseline: both components are maximally simple
function policy_naive_naive(env, portfolio, config, σ)
    return symmetric_naive_policy(env, portfolio, config, σ)
end

# Named policy registry (for figures and tables)
const POLICIES = [
    (fn = policy_glft_ww,    name = "GLF-T + WW",       color = :steelblue,   ls = :solid),
    (fn = policy_glft_naive, name = "GLF-T + Naive",    color = :darkorange,  ls = :dash),
    (fn = policy_naive_ww,   name = "Naive + WW",       color = :mediumseagreen, ls = :dot),
    (fn = policy_naive_naive,name = "Naive + Naive",    color = :crimson,     ls = :dashdot),
]

# ============================================================
# Section 4: Rich Episode Runner
# Runs a single episode and returns per-step StepInfo alongside
# auxiliary time-series needed for figures.
# ============================================================

struct EpisodeData
    total_reward::Float64
    steps::Vector{StepInfo}           # one StepInfo per timestep
    spread_values::Vector{Float64}    # config.spread_levels[action.spread_idx] per step
    hedge_traded::Vector{Bool}        # true if hedge action ≠ :no_trade
    net_Δ_pre::Vector{Float64}        # net_Δ before step (agent_state.net_Δ)
    regime_history::Vector{Int}       # true regime index per step (from vol_state)
    τ_history::Vector{Float64}        # time-to-expiry per step (years)
    cumulative_reward::Vector{Float64}
end

function run_episode(
    policy_fn,
    vol_model::VolModel,
    config::SimConfig,
    rng::AbstractRNG;
    level::Int
)::EpisodeData
    # Build fresh environment
    env = EnvironmentState(
        AgentState(config.S0, config.T_option * config.Δt,
                   0.0, 0.0, 0.0, 0.0,
                   fill(1.0 / length(vol_model.σ_levels), length(vol_model.σ_levels))),
        VolState(vol_model),
        OptionContract[OptionContract(round(config.S0), true)],
        0
    )
    portfolio = Portfolio()
    push!(portfolio.option_quantities, 0)
    initialize_episode!(env, portfolio, vol_model, config; level = level)

    steps            = StepInfo[]
    spread_values    = Float64[]
    hedge_traded     = Bool[]
    net_Δ_pre        = Float64[]
    regime_history   = Int[]
    τ_history        = Float64[]
    cumulative_reward = Float64[]
    total_reward     = 0.0
    done             = false

    while !done
        σ = oracle_σ(env)
        action = policy_fn(env, portfolio, config, σ)

        # Record state before step
        push!(spread_values,  config.spread_levels[action.spread_idx])
        push!(hedge_traded,   config.Δ_targets[action.hedge_idx] != :no_trade)
        push!(net_Δ_pre,      env.agent_state.net_Δ)
        push!(regime_history, env.vol_state.regime_idx)
        push!(τ_history,      env.agent_state.τ)

        next_state, reward, done, step_info = step_environment!(
            env, portfolio, action, config, rng; level = level
        )

        push!(steps, step_info)
        total_reward += reward
        push!(cumulative_reward, total_reward)
    end

    return EpisodeData(
        total_reward, steps,
        spread_values, hedge_traded, net_Δ_pre,
        regime_history, τ_history, cumulative_reward
    )
end

# ============================================================
# Section 5: Monte Carlo Evaluation
# Runs n_episodes and aggregates summary statistics +
# the per-step data needed for figure 2 (spread vs τ) and
# figure 3 (hedge behavior).
# ============================================================

struct PolicyResults
    name::String
    episode_rewards::Vector{Float64}
    mean_reward::Float64
    std_reward::Float64
    sharpe::Float64
    mean_spread::Float64                   # mean chosen half-spread ($) across all steps
    hedge_freq::Float64                    # fraction of steps where a hedge trade was made
    mean_abs_net_Δ::Float64                # mean |net_Δ| across all steps
    mean_hedge_cost_per_episode::Float64   # mean total transaction cost per episode

    # Per-step aggregated series (for plotting)
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
    level::Int
)::PolicyResults
    episode_rewards     = Float64[]
    episode_hedge_costs = Float64[]   # total transaction cost per episode
    all_spread          = Float64[]
    all_hedge_traded    = Bool[]
    all_abs_net_Δ       = Float64[]
    all_τ               = Float64[]
    all_spread_for_τ    = Float64[]

    for _ in 1:n_episodes
        ep = run_episode(policy_fn, vol_model, config, rng; level = level)
        push!(episode_rewards,     ep.total_reward)
        push!(episode_hedge_costs, sum(s.hedge_cost for s in ep.steps))
        append!(all_spread,        ep.spread_values)
        append!(all_hedge_traded,  ep.hedge_traded)
        append!(all_abs_net_Δ,     abs.(ep.net_Δ_pre))
        append!(all_τ,             ep.τ_history)
        append!(all_spread_for_τ,  ep.spread_values)
    end

    μ = mean(episode_rewards)
    σ = std(episode_rewards)
    sharpe = σ > 1e-10 ? μ / σ : 0.0

    # Bin spread by τ (time-to-expiry in trading days, rounded)
    τ_in_days = round.(all_τ .* 252)
    τ_unique  = sort(unique(τ_in_days))
    mean_spread_by_τ = [mean(all_spread_for_τ[τ_in_days .== t]) for t in τ_unique]

    return PolicyResults(
        name, episode_rewards, μ, σ, sharpe,
        mean(all_spread),
        mean(all_hedge_traded),
        mean(all_abs_net_Δ),
        mean(episode_hedge_costs),
        Float64.(τ_unique),
        mean_spread_by_τ
    )
end

# ============================================================
# Section 6: Figure Generators
# ============================================================

# --- Figure 1: P&L Distribution Histograms ---
# Two panels (const vol, Hardy) × 4 overlapping policies
function fig_pnl_distributions(
    results_const::Vector{PolicyResults},
    results_hardy::Vector{PolicyResults};
    save_path::String = "results/fig1_pnl_distributions.png"
)
    p1 = plot(title = "Constant Volatility (σ = 0.20)",
              xlabel = "Episode P&L", ylabel = "Frequency",
              legend = :topright, titlefontsize = 10)

    p2 = plot(title = "Regime-Switching Volatility (Hardy 2001)",
              xlabel = "Episode P&L", ylabel = "Frequency",
              legend = :topright, titlefontsize = 10)

    for (i, pol) in enumerate(POLICIES)
        r_const = results_const[i]
        r_hardy = results_hardy[i]
        bins = 40

        histogram!(p1, r_const.episode_rewards,
                   bins = bins, alpha = 0.4,
                   label = pol.name, color = pol.color,
                   normalize = :probability)

        histogram!(p2, r_hardy.episode_rewards,
                   bins = bins, alpha = 0.4,
                   label = pol.name, color = pol.color,
                   normalize = :probability)
    end

    # Annotate Sharpe ratios on each panel
    for (i, pol) in enumerate(POLICIES)
        annotate!(p1, :topright, text("", 1))   # placeholder — Sharpe in table
    end

    fig = plot(p1, p2, layout = (1, 2), size = (900, 380),
               margin = 5mm, dpi = 150)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

# --- Figure 2: Mean Spread vs Time-to-Expiry ---
# Shows GLF-T widening as γ rises near expiry; naive stays flat
function fig_spread_vs_tau(
    results_const::Vector{PolicyResults},
    results_hardy::Vector{PolicyResults};
    save_path::String = "results/fig2_spread_vs_tau.png"
)
    # xflip=true: high τ (far from expiry) on left, low τ (near expiry) on right.
    # Spread rises rightward as gamma peaks approaching expiry — correct reading direction.
    p1 = plot(title = "Constant Volatility",
              xlabel = "Days to Expiry", ylabel = "Mean Half-Spread (\$)",
              legend = :topright, titlefontsize = 10, xflip = true)

    p2 = plot(title = "Regime-Switching Volatility",
              xlabel = "Days to Expiry", ylabel = "Mean Half-Spread (\$)",
              legend = :topright, titlefontsize = 10, xflip = true)

    for (i, pol) in enumerate(POLICIES)
        r_const = results_const[i]
        r_hardy = results_hardy[i]

        plot!(p1, r_const.τ_buckets, r_const.mean_spread_by_τ,
              label = pol.name, color = pol.color,
              linestyle = pol.ls, linewidth = 2)

        plot!(p2, r_hardy.τ_buckets, r_hardy.mean_spread_by_τ,
              label = pol.name, color = pol.color,
              linestyle = pol.ls, linewidth = 2)
    end

    fig = plot(p1, p2, layout = (1, 2), size = (900, 380),
               margin = 5mm, dpi = 150)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

# --- Figure 3: Hedge Behavior — frequency and mean |net Δ| ---
function fig_hedge_behavior(
    results_const::Vector{PolicyResults},
    results_hardy::Vector{PolicyResults};
    save_path::String = "results/fig3_hedge_behavior.png"
)
    names       = [pol.name for pol in POLICIES]
    colors      = [pol.color for pol in POLICIES]
    hedge_const = [r.hedge_freq for r in results_const]
    hedge_hardy = [r.hedge_freq for r in results_hardy]
    Δ_const     = [r.mean_abs_net_Δ for r in results_const]
    Δ_hardy     = [r.mean_abs_net_Δ for r in results_hardy]

    # Grouped bar charts
    p1 = groupedbar(
        hcat(hedge_const, hedge_hardy),
        bar_position = :dodge,
        bar_width = 0.6,
        xticks = (1:4, names),
        xrotation = 15,
        ylabel = "Hedge Trade Frequency",
        title = "Hedge Frequency",
        label = ["Const Vol" "Hardy"],
        color = [:steelblue :darkorange],
        titlefontsize = 10
    )

    p2 = groupedbar(
        hcat(Δ_const, Δ_hardy),
        bar_position = :dodge,
        bar_width = 0.6,
        xticks = (1:4, names),
        xrotation = 15,
        ylabel = "Mean |net Δ|",
        title = "Mean Absolute Net Delta",
        label = ["Const Vol" "Hardy"],
        color = [:steelblue :darkorange],
        titlefontsize = 10
    )

    fig = plot(p1, p2, layout = (1, 2), size = (900, 400),
               margin = 8mm, dpi = 150)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

# --- Figure 4: Cumulative P&L Trace (single representative episode) ---
# All 4 policies run on the same market path (same rng seed).
# Regime switches shaded for the Hardy environment.
function fig_cumulative_pnl(
    vol_model::VolModel,
    config::SimConfig;
    level::Int,
    env_name::String,
    seed::Int = 1234,
    save_path::String = "results/fig4_cumulative_pnl.png"
)
    p = plot(title = "Cumulative P&L — $env_name (single episode)",
             xlabel = "Timestep", ylabel = "Cumulative P&L",
             legend = :topleft, titlefontsize = 10)

    # --- Step 1: draw regime shading first (Hardy only), before any policy lines ---
    # Use the first policy's episode to get regime history (same seed = same path)
    if length(vol_model.σ_levels) > 1
        ep_for_regimes = run_episode(
            POLICIES[1].fn, vol_model, config, MersenneTwister(seed); level = level
        )
        shaded_first = false
        in_high_vol  = false
        region_start = 0
        for (t, reg) in enumerate(ep_for_regimes.regime_history)
            if reg == 2 && !in_high_vol
                region_start = t
                in_high_vol  = true
            elseif reg != 2 && in_high_vol
                lbl = shaded_first ? "" : "High-Vol Regime"
                vspan!(p, region_start, t - 1,
                       alpha = 0.12, color = :crimson, label = isempty(lbl) ? nothing : lbl)
                shaded_first = true
                in_high_vol  = false
            end
        end
        if in_high_vol
            lbl = shaded_first ? "" : "High-Vol Regime"
            vspan!(p, region_start, length(ep_for_regimes.regime_history),
                   alpha = 0.12, color = :crimson, label = isempty(lbl) ? nothing : lbl)
        end
    end

    # --- Step 2: plot all 4 policy cumulative P&L traces ---
    for pol in POLICIES
        ep = run_episode(pol.fn, vol_model, config, MersenneTwister(seed); level = level)
        plot!(p, ep.cumulative_reward,
              label = pol.name, color = pol.color,
              linestyle = pol.ls, linewidth = 1.5)
    end

    hline!(p, [0.0], color = :black, linestyle = :dash, linewidth = 0.8, label = "")

    fig = plot(p, size = (800, 380), margin = 5mm, dpi = 150)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

# ============================================================
# Section 7: Summary Table
# Prints to stdout and saves as a plain-text file for the paper
# ============================================================

function print_summary_table(
    results_const::Vector{PolicyResults},
    results_hardy::Vector{PolicyResults};
    save_path::String = "results/table1_summary.txt"
)
    mkpath(dirname(save_path))
    lines = String[]

    header = @sprintf("%-20s  %10s  %10s  %10s  %10s  %10s  %10s  %12s",
                      "Policy",
                      "Mean P&L", "Std P&L", "Sharpe",
                      "Spread(\$)", "Hedge%", "|net Δ|", "Hedge Cost")
    divider = "─"^96

    function table_block(results, env_label)
        push!(lines, "")
        push!(lines, "Environment: $env_label")
        push!(lines, divider)
        push!(lines, header)
        push!(lines, divider)
        for r in results
            push!(lines, @sprintf(
                "%-20s  %10.4f  %10.4f  %10.4f  %10.4f  %9.1f%%  %10.4f  %12.4f",
                r.name,
                r.mean_reward,
                r.std_reward,
                r.sharpe,
                r.mean_spread,
                r.hedge_freq * 100,
                r.mean_abs_net_Δ,
                r.mean_hedge_cost_per_episode
            ))
        end
        push!(lines, divider)
    end

    table_block(results_const, "Constant Volatility (σ = 0.20)")
    table_block(results_hardy, "Regime-Switching (Hardy 2001: σ₁=0.121, σ₂=0.269)")

    output = join(lines, "\n")
    println(output)

    open(save_path, "w") do f
        println(f, output)
    end
    println("\nSaved: $save_path")
end

# ============================================================
# Section 8: Main Entry Point
# ============================================================

function run_evaluation(;
    n_episodes::Int = 1_000,
    seed::Int = 42
)
    rng = MersenneTwister(seed)

    println("="^60)
    println("Options Market Making — Benchmark Evaluation")
    println("n_episodes = $n_episodes  |  seed = $seed")
    println("="^60)

    # Determine level for each vol model
    level_const = 1   # constant vol: no regime uncertainty
    level_hardy = 2   # regime-switching, known regime (oracle σ)

    # --- Evaluate all 4 policies under both environments ---
    results_const = PolicyResults[]
    results_hardy = PolicyResults[]

    for pol in POLICIES
        println("\nEvaluating: $(pol.name)")

        print("  Constant vol...  ")
        r_const = evaluate_policy(
            pol.fn, pol.name, VM_CONST, SIM_CONFIG,
            n_episodes, rng; level = level_const
        )
        push!(results_const, r_const)
        @printf("done — mean P&L = %.4f, Sharpe = %.4f\n",
                r_const.mean_reward, r_const.sharpe)

        print("  Hardy regime...  ")
        r_hardy = evaluate_policy(
            pol.fn, pol.name, VM_HARDY, SIM_CONFIG,
            n_episodes, rng; level = level_hardy
        )
        push!(results_hardy, r_hardy)
        @printf("done — mean P&L = %.4f, Sharpe = %.4f\n",
                r_hardy.mean_reward, r_hardy.sharpe)
    end

    # --- Produce all figures ---
    println("\nGenerating figures...")
    fig_pnl_distributions(results_const, results_hardy)
    fig_spread_vs_tau(results_const, results_hardy)
    fig_hedge_behavior(results_const, results_hardy)
    fig_cumulative_pnl(VM_CONST, SIM_CONFIG;
                       level = level_const, env_name = "Constant Vol",
                       seed = 77,
                       save_path = "results/fig4a_cumulative_pnl_const.png")
    fig_cumulative_pnl(VM_HARDY, SIM_CONFIG;
                       level = level_hardy, env_name = "Hardy Regime-Switching",
                       save_path = "results/fig4b_cumulative_pnl_hardy.png")

    # --- Print and save summary table ---
    println("\nSummary Table:")
    print_summary_table(results_const, results_hardy)

    println("\nEvaluation complete. All outputs in results/")
    return results_const, results_hardy
end