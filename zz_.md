Project Structure

Julia POMDP/MDP market-making simulator, ~2,600 lines across `src/1_types.jl` through `src/10_pomcpow.jl`, plus 2 test files (71 tests total, all passing).

| File | Lines | What it actually does |
|---|---|---|
| `1_types.jl` | 175 | All structs: `SimConfig`, `VolModel` (validates transition matrix rows sum to 1, computes stationary dist for 1–2 regimes), `MarketMakingAction`, `AgentState`, `ParticleFilter`, `FillOutcome` |
| `2_black_scholes.jl` | 191 | BS price/Δ/Γ/ν/Θ (`bs_price`, `bs_Δ_Γ`, `bs_all`), plus `bs_all_belief_weighted` which prices per-regime then averages (correct Jensen's-inequality handling). One dead stub: `calc_σ_level` (`docs/DMU_Final_Paper.md:189` — body is empty) |
| `3_spot_dynamics.jl` | 33 | `step_spot`: GBM with regime-conditional σ, Markov regime transition sampled from `transition_matrix` row |
| `4_fills.jl` | 55 | Avellaneda-Stoikov exponential fill intensity `λ = A·exp(-k·δ_quote)`, independent Bernoulli bid/ask fills evaluated against true `V_market` (not agent's belief) — this asymmetry is the actual POMDP signal |
| `5_portfolio.jl` | 122 | Portfolio Greeks (`compute_portfolio`), true mark-to-market wealth (`compute_true_wealth`), hedge execution with proportional cost, reward (`compute_reward`) |
| `6_environment.jl` | 140 | `initialize_episode!`, `step_environment!` — the 12-step episode loop (quote → fill vs. true price → hedge → GBM step → regime transition → expiry/rollover → particle filter update → reward) |
| `7_benchmarks.jl` | 229 | Four analytical policies (below) + `run_benchmark` Monte Carlo runner |
| `8_evaluation.jl` | 462 | Runs the four benchmarks (1,000 episodes) + MCTS (same episode count), generates all 7 figures and `table1_summary.txt`. **Does not call POMCPOW** |
| `9_mcts_mdp.jl` | 320 | Full MDP formulation with oracle regime knowledge, MCTS-DPW solver via `MCTS.jl`/`POMDPs.jl` |
| `10_pomcpow.jl` | 394 | POMDP formulation + POMCPOW solver — **broken**, see below |
| `9_value_iteration.jl`, `11_dqn.jl`, `13_qmdp.jl`, `10_pomdp_interface.jl` | 0, 0, 0, 1 | Empty files. No value iteration, no DQN, no QMDP were implemented — these are filenames only |

**README vs. code**: closely aligned — I verified the README's structural claims, formulas, and defaults against the actual code and they match. No drift there.

## POMDP Formulation

**State** (latent + observed): true state is `(S, V*, σ*, q_options, q_spot, cash, τ)`. `σ*` (the Hardy regime) is latent to the agent; everything else is fully observed.

**Belief representation — this is the key nuance**: two different filters exist for two different purposes, and it's easy to conflate them:
- `12_belief_updater.jl`'s `ParticleFilter` is a **real bootstrap particle filter** over continuous σ (log-normal prior, GBM log-likelihood weighting, systematic resampling + jitter when ESS < n/2). This is used by the analytical benchmarks and the environment's default path.
- `10_pomcpow.jl` separately uses `ParticleFilters.jl`'s `WeightedParticleBelief` inside the POMCPOW tree, with Rao-Blackwellization (fully-observable quantities like S, q, τ, cash are synced from ground truth each step; only `σ_particle` stays uncertain).

**Action space**: continuous, exactly as documented — `MarketMakingAction(δ, Δ_target)`, no discretization anywhere in the codebase.

**Reward** (`5_portfolio.jl:70-77`, mirrored in `9_mcts_mdp.jl:130` and `10_pomcpow.jl:136`): exactly two terms —
```
r_t = (wealth_after − wealth_before) − φ · Δ_target²
```
`wealth_after/before` already nets spread capture, hedge P&L, and hedge transaction cost (`κ·|shares|·S`) via `compute_true_wealth`/`execute_hedge!`. There is no separate transaction-cost term in the reward — it's absorbed into the wealth delta. Confirmed: no aspirational/unimplemented reward terms.

## Simulation Environment

Julia, `Project.toml` deps confirmed: `POMDPs`, `POMCPOW`, `POMDPTools`, `BasicPOMCP`, `MCTS`, `ParticleFilters`, `Distributions`, `Flux` (listed but unused — no DQN code exists), `Plots`/`StatsPlots`, `StatsBase`.

- **GBM**: `3_spot_dynamics.jl:17`
- **Regime-switching (Hardy 2001, 2-state Markov)**: `VolModel`/`VolState` in `1_types.jl`, transition sampling in `3_spot_dynamics.jl:24-27`
- **Poisson-style fills (Avellaneda-Stoikov exponential intensity, Bernoulli realization — not a literal Poisson process, but the standard AS fill model)**: `4_fills.jl:11-42`

## Policies Benchmarked

| Policy | Status |
|---|---|
| GLF-T + Whalley-Wilmott | Fully implemented, `7_benchmarks.jl:77-100` |
| GLF-T + Naive hedge | Fully implemented |
| Naive spread + WW | Fully implemented |
| Naive + Naive | Fully implemented |
| MCTS-DPW on oracle MDP | Fully implemented and **runs successfully** (`9_mcts_mdp.jl`) — this is search/planning, not a learned RL policy (no function approximation, replans from scratch each step via `solve(solver, mdp)`)
|
| POMCPOW | Implemented but **non-functional** — I ran it directly and it throws immediately: `MethodError: no method matching step_environment!(...; σ_hat_override::Float64)`. The function calls a keyword argument (`σ_hat_override`) that was never added to `step_environment!`'s actual signature in `6_environment.jl` (only `oracle_regime` exists). This matches commit `2019929 "work from claude code. Doesn't really work"`. `8_evaluation.jl` does not invoke it — consistent with it being known-broken and excluded from the pipeline. |
| DQN, QMDP, Value Iteration | Not implemented — files are 0 bytes |

## Results (verified from `results/table1_summary.txt` + `full_eval_output.txt`, 1,000 episodes)

**Constant vol (σ=0.20):**
| Policy | Mean P&L | Std | Sharpe |
|---|---|---|---|
| GLF-T+WW | 3.971 | 2.244 | 1.770 |
| GLF-T+Naive | 3.640 | 1.930 | 1.886 |
| Naive+WW | 4.625 | 4.054 | 1.141 |
| Naive+Naive | 4.165 | 3.430 | 1.215 |

**Hardy regime-switching:**
| Policy | Mean P&L | Std | Sharpe |
|---|---|---|---|
| GLF-T+WW | 4.104 | 2.319 | 1.770 |
| GLF-T+Naive | 3.684 | 2.001 | 1.841 |
| Naive+WW | 4.306 | 4.388 | 0.981 |
| Naive+Naive | 3.869 | 3.799 | 1.018 |

MCTS-DPW (Hardy, oracle, 1,000 episodes): mean 2.734, std 2.802, Sharpe 0.976.

**Correcting your two claims:**
1. **"GLFT reduced P&L std by ~50% vs. naive fixed-spread"** — directionally right but overstated. Actual reduction ranges from ~35% (GLF-T+WW std 2.24 vs. Naive+Naive std 3.43, constant vol) to ~47% (GLF-T+WW std 2.32 vs. Naive+WW std 4.39, Hardy). "~40-45%" is more defensible than "~50%"; don't round up to 50%.
2. **"RL underperformed GLFT across tested configurations"** — true only for MCTS-DPW (Sharpe 0.98 vs. best GLFT Sharpe 1.84-1.89), and MCTS is search/planning with an oracle, not a trained RL agent. No DQN or trained RL policy was ever evaluated (files are empty), and POMCPOW never produced a result at all (crashes). I'd phrase the bullet as **"MCTS-based planning underperformed analytical (GLFT) benchmarks despite oracle state access"** rather than "RL" broadly — "RL" implies a learned policy that doesn't exist in this codebase yet.

## Explicitly NOT implemented / scaffolding only
- Value iteration (`9_value_iteration.jl` — empty)
- DQN (`11_dqn.jl` — empty, despite Flux being a listed dependency)
- QMDP (`13_qmdp.jl` — empty)
- POMCPOW evaluation (`10_pomcpow.jl` — code exists, crashes on invocation, never produced a number)
- `calc_σ_level` in `2_black_scholes.jl` — empty function body

The paper draft (`docs/DMU_Final_Paper.md:229`) itself correctly frames the POMDP/POMCPOW work as future work ("the natural next step"), not a completed result — so the paper and code are honest with each other here; it's your resume-bullet framing that needs to match that, not overclaim POMCPOW/RL as delivered.

## Tests
`tests/test_1_2_3.jl` (42 tests: types, BS pricing incl. Jensen's inequality check, GBM mean/std/regime-fraction checks) and `tests/test_4.jl` (29 tests: fill probability encoding, portfolio updates from fills, action bound sanity). I ran both directly — 71/71 pass. No tests exist for MCTS, POMCPOW, or the evaluation/figure-generation code.