# Implementation Plan: Options Market Making Under Uncertainty

## Status Overview (April 13, 2026)

**Modules 1–8 complete.** Derivatives paper due April 17 — writing phase starts now. DMU paper due April 30 — RL modules start April 18.

---

## Completed Modules

### Module 1: Types and Configuration — `1_types.jl` ✓

Core structs: `SimConfig`, `VolModel`, `VolState`, `OptionContract`, `AgentState`, `EnvironmentState`, `MarketMakingAction`, `Portfolio`, `FillOutcome`, `StepInfo`.

**Current SimConfig defaults:**
```julia
S0=100.0, r=0.05, Δt=1/252
T_option=63, n_options_per_episode=8   # 504 steps/episode
κ=0.001, A=140.0, k=6.0
γ_market=0.1, φ=0.01
spread_levels = [0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
Δ_targets = [:no_trade, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0,
              0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
```

n_actions = 6 × 14 = 84.

`StepInfo` is defined here and returned by `step_environment!`:
```julia
struct StepInfo
    log_return::Float64
    fill::FillOutcome
    shares_traded::Float64
    hedge_cost::Float64
    wealth_before::Float64
    wealth_after::Float64
end
```

`perfect_regime_belief(vs)` returns `vs.vm.transition_matrix[vs.regime_idx, :]` — used by the market, by oracle σ, and by Level 2 belief initialization.

---

### Module 2: Black-Scholes — `2_black_scholes.jl` ✓

`bs_all(S, K, τ, σ, r; call)` → `(; price, Δ, Γ, ν, Θ)`. Theta computed analytically. Guard clauses at τ≈0.

`bs_all_belief_weighted(S, K, τ, σ_regimes, beliefs, r; call)` → belief-weighted price and Greeks. Weights each regime's BS output separately (Jensen's inequality: must price per-regime then average, not average σ then price).

---

### Module 3: Spot Dynamics — `3_spot_dynamics.jl` ✓

`step_spot(S, vs, config, rng)` → `(S_new, vs_new, log_return)`. GBM with risk-neutral drift, regime transition via Markov chain.

---

### Module 4: Fill Model — `4_fills.jl` ✓

AS (2008) fill intensity: `fill_probability(δ, A, k, Δt) = A·exp(-k·δ)·Δt`.

`simulate_fills(bid_price, ask_price, V_market, config, rng)` → `FillOutcome`. Fill probs computed against V_market; agent quotes around V_believed. In Levels 1–2, V_believed = V_market so fills are symmetric.

`fill_outcome_likelihood(outcome, V_market_j, config)` — used by Hamilton filter in Level 3.

---

### Module 5: Portfolio — `5_portfolio.jl` ✓

`compute_portfolio(portfolio, options, S, τ, σ_regimes, beliefs, r)` → `(; portfolio_value, Δ_net, Δ_options, Γ_net, ν_net, Θ_net)`.

`execute_hedge!(portfolio, target_Δ, current_net_Δ, S, config)` → `(shares_traded, hedge_cost)`. Charges κ·|shares|·S.

`reset_for_new_option!(portfolio, options, S)` — settles at intrinsic, starts new ATM call at K=round(S).

`compute_reward(wealth_before, wealth_after, net_Δ, config)` → `pnl - φ·net_Δ²`.

---

### Module 6: Environment — `6_environment.jl` ✓

`initialize_episode!(env, portfolio, vol_model, config; level)` — initializes regime belief based on level: `[1.0]` for L1, one-hot true regime for L2, uniform for L3.

`step_environment!(env, portfolio, action, config, rng; belief_update_fn, level)` → `(next_state, reward, done, StepInfo)`.

**StepInfo is the 4th return value** (changed from `log_return, fill` as separate values). All 6 fields are computed internally and packed before return.

`update_belief` router dispatches to one-hot (Level 2) or `belief_update_fn` (Level 3, Module 12).

---

### Module 7: Benchmarks — `7_benchmarks.jl` ✓

**Spread formulas (options-adapted):**

GLF-T half-spread: `δ* = γ·|Γ|·S²·σ²·τ + (2/γ)·ln(1 + γ/k)`

AS half-spread: `δ* = (1/k)·ln(1 + γ/k) + γ·|Γ|·S²·σ²·τ`

Dollar-gamma substitution γσ²τ → γ·Γ·S²·σ²·τ is the key adaptation for options.

**Hedge formulas:**

WW band halfwidth: `H = (3κ/(2φ) · Γ²S²σ²)^(1/3) · Δt^(1/3)`

When |net_Δ| ≤ H: `:no_trade`. When outside: trade to band edge `sign(net_Δ) × H`.

Naive hedge: always target net_Δ = 0.

**`run_benchmark(policy_fn, σ_fn, vol_model, config, n_episodes, rng; level)`:** Summary statistics runner (no per-step StepInfo collection). Used for quick validation; Module 8's `run_episode` is the richer evaluation harness.

**Important:** `run_benchmark` unpacks `step_environment!` as `next_state, reward, done, _ = step_environment!(...)` (4 values, not 5).

---

### Module 8: Evaluation — `8_evaluation.jl` ✓

**Four policies:**
1. `policy_glft_ww` — GLF-T spread + WW hedge
2. `policy_glft_naive` — GLF-T spread + naive hedge (target net_Δ = 0)
3. `policy_naive_ww` — fixed $0.10 spread + WW hedge
4. `policy_naive_naive` — fixed $0.10 spread + naive hedge

**Oracle σ function:**
```julia
function oracle_σ(env)
    weights = perfect_regime_belief(env.vol_state)
    return sqrt(sum(weights .* env.vol_state.vm.σ_levels .^ 2))
end
```
Transition-row-weighted variance-equivalent σ. Gives benchmarks same information market uses. Consistent with Level 2 RL.

**Two vol environments:**
- `VM_CONST = VolModel([0.20])` — single regime, variance-weighted Hardy average
- `VM_HARDY = VolModel([0.121, 0.269], transition_matrix=[0.9982 0.0018; 0.0022 0.9978])`

**Level assignments:**
- Constant vol → `level=1`
- Hardy regime-switching → `level=2`

**`EpisodeData` struct** collects per-step: `StepInfo[]`, spread values, hedge traded flags, net_Δ pre-step, regime history, τ history, cumulative reward.

**`PolicyResults` struct** aggregates across n_episodes: mean/std/Sharpe P&L, mean spread, hedge frequency, mean |net_Δ|, mean hedge cost per episode, τ-bucketed spread series.

**Hedge cost** sourced from `sum(s.hedge_cost for s in ep.steps)` using `StepInfo.hedge_cost`.

**Figures produced:**
- `fig1_pnl_distributions.png` — histogram of episode P&L, 4 policies × 2 envs
- `fig2_spread_vs_tau.png` — mean spread vs days to expiry (xflip=true so time flows left→right)
- `fig3_hedge_behavior.png` — hedge frequency and |net_Δ| bar charts (StatsPlots groupedbar)
- `fig4a_cumulative_pnl_const.png` — single episode trace, constant vol
- `fig4b_cumulative_pnl_hardy.png` — single episode trace, Hardy with regime shading
- `results/table1_summary.txt` — full results table

**Known cosmetic issue:** `vspan!` legend in fig4b still shows repeated "High-Vol Regime" entries in some Plots backends despite `label=nothing` fix. Non-blocking for paper submission.

**Entry point:** `run_evaluation(n_episodes=1_000)` in `scripts/run_evaluation.jl`.

---

## Pending Modules (April 18–30)

### Module 9: Value Iteration — `9_value_iteration.jl`

Tabular VI for Level 1. Discretized state space over (net_Δ, moneyness, τ). Bellman backup until convergence. Target: April 18.

Key decision: state discretization granularity. Suggested grid: net_Δ in [-0.3, 0.3] step 0.05 (13 values), moneyness in [0.85, 1.15] step 0.05 (7 values), τ in {1, 8, 16, 24, 32, 40, 48, 56, 63} days (9 values). Total states: ~800.

Expected result: learned policy should approximately recover GLF-T spread behavior and WW band behavior, validating the simulation.

---

### Module 10: POMDPs.jl Interface — `10_pomdp_interface.jl`

Interface to DPWSolver (MCTS, Level 2) and POMCPOW (Level 3). Implements `POMDPs.jl` required functions: `gen`, `initialstate`, `reward`, `discount`, `actions`, `obstype`. Target: April 19.

---

### Module 11: DQN — `11_dqn.jl`

3-layer MLP, replay buffer, ε-greedy, target network. From scratch with Flux.jl. Level 2 only. Regime index as one-hot feature. Target: April 20.

If DQN doesn't converge: fall back to MCTS-only for Level 2.

---

### Module 12: Hamilton Filter — `12_belief_updater.jl`

Level 3 only. Four-step update:
1. Predict: ξ_predict = P' × ξ_prior
2. Return likelihood: η_j = N(log_return | r·Δt, σⱼ√Δt)
3. Fill likelihood: ℓ_j = P(fill_outcome | regime=j) via `fill_outcome_likelihood`
4. Normalize: ξ_new = (ξ_predict ⊙ η ⊙ ℓ) / sum(...)

Returns `belief_update_fn` compatible with `update_belief` router in Module 6. Target: April 22.

---

### Module 13: QMDP — `13_qmdp.jl`

Solve MDP per regime via VI (reuses Module 9). Weight Q-values by Hamilton filter belief: `a* = argmax_a Σⱼ belief_j · Q_j(s,a)`. QMDP assumption: full observability at next step. Useful as lower bound on POMDP performance. Target: April 23.

---

## Build Order (April 18–30)

| Day | Task |
|---|---|
| Apr 15–17 | Write and submit derivatives paper |
| Apr 18 | Module 9: Value iteration |
| Apr 19 | Module 10: POMDPs.jl interface (MCTS + POMCPOW shell) |
| Apr 20 | Module 11: DQN |
| Apr 21 | Run Level 1 and Level 2 evaluations, collect results |
| Apr 22 | Module 12: Hamilton filter (belief updater) |
| Apr 23 | Module 13: QMDP |
| Apr 24 | POMCPOW integration, run Level 3 evaluation |
| Apr 25–30 | Write and submit DMU paper |

---

## Risk Mitigation

**If value iteration is too slow:** Reduce state discretization first.

**If DQN doesn't converge:** Report MCTS only for Level 2. DQN convergence in financial environments is known to be unstable (confirmed by referenced papers).

**If POMCPOW is too slow:** Report QMDP only for Level 3, note POMCPOW as future work.

**If Level 3 belief updater is too complex to finish:** Skip POMCPOW, report QMDP + cost-of-partial-observability analysis using Level 2 vs Level 3 QMDP.

**Minimum viable DMU paper:** Level 1 VI + Level 2 MCTS (no DQN, no POMDP). Still covers MDP formulation, value iteration, online tree search — sufficient for the DMU course.
