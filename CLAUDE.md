# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Options market-making POMDP implemented in Julia. A market maker quotes bids/asks on a single-strike ATM call, delta-hedges the underlying, and jointly optimizes spread width and hedge ratio. Dual submission: a derivatives finance paper (analytical benchmarks) and a DMU/RL paper (POMDP solvers).

## Running Code

```bash
# Run tests
julia tests/test_1_2_3.jl     # Types, Black-Scholes, spot dynamics
julia tests/test_4.jl          # Fill model

# Run the full benchmark evaluation (1,000 episodes, produces figures + table)
julia scripts/run_evaluation.jl

# Run benchmark script directly
julia scripts/run_benchmarks.jl
```

There is no build step or package compilation. Each file uses `include()` to load dependencies. Modules must be loaded in numeric order because each `include`s the ones before it.

## Module Architecture

Files in `src/` are numbered to indicate load order and dependency:

| Module | File | Role |
|--------|------|------|
| 1 | `1_types.jl` | All structs: `SimConfig`, `VolModel`, `VolState`, `AgentState`, `EnvironmentState`, `Portfolio`, `MarketMakingAction`, `StepInfo`, `FillOutcome`, `OptionContract` |
| 2 | `2_black_scholes.jl` | `bs_all`, `bs_all_belief_weighted`, `bs_price` — belief-weighted BS pricing across regimes |
| 3 | `3_spot_dynamics.jl` | `step_spot` — GBM with Markov regime switching |
| 4 | `4_fills.jl` | `compute_quotes`, `simulate_fills`, `update_from_fills!` — AS (2008) fill model |
| 5 | `5_portfolio.jl` | `compute_portfolio`, `execute_hedge!`, `compute_reward` |
| 6 | `6_environment.jl` | `initialize_episode!`, `step_environment!` — main simulation loop |
| 7 | `7_benchmarks.jl` | Analytical policies: `glft_ww_policy`, `symmetric_naive_policy`, `run_benchmark` |
| 8 | `8_evaluation.jl` | `run_evaluation` — Monte Carlo evaluation across 4 policies × 2 environments + figure generation |
| 9–13 | `9_value_iteration.jl` … `13_qmdp.jl` | RL solvers (stubs — pending implementation) |

Each `src/` file includes its own dependencies at the top via `include()`. The scripts in `scripts/` simply `include` the highest-numbered module they need.

`sandbox.jl` at the project root is a scratch/draft file used during development.

## Key Abstractions

**Action space:** The action space is currently implemented as 6 spread levels × 14 hedge targets = 84 discrete actions. `MarketMakingAction` holds `(spread_idx, hedge_idx)`. `Δ_targets[1]` is always `:no_trade`; indices 2–14 are numeric delta targets from −0.3 to +0.3. However, after new developments, the new approach is to have the action space be continuous with bounds: a_t = (\delta_t, \Delta^{\text{target}}_{P,t})
\delta_t \in [c\cdot S_t \cdot |\hat \Delta_t|, \hat V_t]
\Delta^{\text{target}}_{P,t} \in [\text{min}(0,\hat \Delta_{P,t}), \text{max}(0, \hat \Delta_{P, t})]

**`step_environment!`** returns `(next_state, reward, done, StepInfo)`. Internally: quote → fill → hedge → spot step → option expiry check → belief update → reward.

**Belief-weighted pricing:** The market prices options as `Σⱼ P(regime_{t+1}=j | current regime) × V_BS(σⱼ)`. Agents at Level 3 use a Hamilton filter approximation of this. Never use `V_BS(mean_σ)` — Jensen's inequality means `E[V(σ)] ≥ V(E[σ])` for convex BS.

**`perfect_regime_belief(vs::VolState)`** returns the transition matrix row for the current regime — this is what the market uses for pricing, and what benchmark oracle σ is derived from.

**WW hedge convention:** When `|net_Δ| > H`, trade to the band edge `sign(net_Δ) × H`, not to zero. Index 1 in `Δ_targets` is `:no_trade`.

**Reward:** `r_t = ΔWealth_t - φ · net_Δ²` — a soft delta penalty, not hard inventory limits.

## Three Simulation Levels

| Level | Agent info | Belief | Planned solvers |
|-------|-----------|--------|----------------|
| L1 | Constant vol | `[1.0]` | Value Iteration (Module 9) |
| L2 | True regime known | One-hot | MCTS/DPWSolver, DQN (Modules 10–11) |
| L3 | Returns + fills only | Hamilton filter | QMDP, POMCPOW (Modules 12–13) |

Pass `level=` to `initialize_episode!` and `step_environment!`. The `belief_update_fn` argument to `step_environment!` is `nothing` for L1/L2 and a Hamilton filter function for L3.

## Hardy (2001) Regime Parameters

```julia
vm = VolModel(
    [0.121, 0.269],  # σ₁ (low), σ₂ (high)
    transition_matrix = [0.9982 0.0018; 0.0022 0.9978]
)
# Stationary dist: π₁ ≈ 0.55, π₂ ≈ 0.45
# Oracle σ = sqrt(Σⱼ pᵢⱼ σⱼ²)  ← transition-row-weighted variance equivalent
```

Constant-vol baseline uses `VolModel([0.20])` (single regime, `transition_matrix = ones(1,1)`).
