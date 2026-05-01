# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Options market-making POMDP implemented in Julia. A market maker quotes bids/asks on a single-strike ATM call, delta-hedges the underlying, and jointly optimizes spread width and hedge ratio under regime uncertainty. We are working on the DMU Final Paper — it is the single source of truth for all formulations and must be kept in sync with code as modules are added.

## Running Code

```bash
# Always run with --project=. so Julia finds StatsBase and other dependencies
julia --project=. tests/test_1_2_3.jl     # Types, Black-Scholes, spot dynamics (42 tests)
julia --project=. tests/test_4.jl          # Fill model (29 tests)

# Run the full benchmark evaluation (produces figures + table in results/)
julia --project=. src/8_evaluation.jl

# Or invoke run_evaluation() interactively from the REPL
```

No build step. Each file uses `include()` to load dependencies. Modules load in numeric order.

## Module Architecture

| Module | File | Role | Status |
|--------|------|------|--------|
| 1 | `1_types.jl` | All structs | Done |
| 2 | `2_black_scholes.jl` | BS pricing and Greeks | Done |
| 3 | `3_spot_dynamics.jl` | GBM with Markov regime switching | Done |
| 4 | `4_fills.jl` | Fill model, quote computation | Done |
| 5 | `5_portfolio.jl` | Portfolio accounting, reward, hedge | Done |
| 12 | `12_belief_updater.jl` | Particle filter for σ* inference | Done |
| 6 | `6_environment.jl` | Episode runner: `initialize_episode!`, `step_environment!` | Done |
| 7 | `7_benchmarks.jl` | Analytical policies: GLF-T+WW, GLF-T+Naive, Naive+WW, Naive+Naive | Done |
| 8 | `8_evaluation.jl` | Monte Carlo evaluation, figure generation | Done |
| 9 | `9_value_iteration.jl` | Value iteration (constant vol baseline) | Pending |
| 10–13 | … | MCTS, DQN, POMCPOW, QMDP solvers | Pending |

Module 12 is numbered out of order because the particle filter must be included before Module 6.
Include order in `7_benchmarks.jl`: `1→2→3→4→5→12→6`.

## Key Abstractions

**Action space (continuous):**
```
a_t = (δ_t, Δ_target_t)
δ_t       ∈ [κ·S·|hat_Δ|,  hat_V]          # half-spread, economic lower bound
Δ_target_t ∈ [min(0, hat_Δ_P), max(0, hat_Δ_P)]  # target net portfolio delta
```
`MarketMakingAction` is `struct MarketMakingAction; δ::Float64; Δ_target::Float64; end`.
No discrete grids — solvers must output these two floats directly.

**Belief (particle filter over σ*):**
- Agent treats σ* as an unknown continuous quantity (does NOT know Hardy parameters)
- `ParticleFilter(n)` holds `log_σ::Vector{Float32}` and `log_w::Vector{Float64}`
- Prior: log-normal centered at log(0.20), std 0.5 in log-space
- `update!(pf, r_t, config)`: weights via GBM log-likelihood; systematic resample + jitter when ESS < n/2
- `get_σ_hat(pf)` returns the weighted-mean σ estimate → goes into `AgentState.σ_hat`
- `ParticleFilter` is NOT inside `AgentState`; it is a separate argument wherever needed

**Observation / `AgentState`:**
```julia
struct AgentState
    r_t::Float64      # log-return this step
    hat_Δ_P::Float64  # net portfolio delta (q·hat_Δ + q_spot)
    hat_Γ_P::Float64  # net portfolio gamma
    f_t::Int          # fill signal: +1 bid only, -1 ask only, 0 otherwise
    τ::Float64        # time to expiry (years)
    σ_hat::Float64    # particle-filter estimate of σ*
    hat_V::Float64    # agent's BS price using σ_hat
    hat_Δ::Float64    # agent's BS delta using σ_hat (for action bounds)
    S::Float64        # current spot (for action bound κ·S·|hat_Δ|)
end
```

**True wealth / reward:**
- Market prices using `perfect_regime_belief(vs)` + `bs_all_belief_weighted` (transition-row-weighted V*)
- Reward: `r_t = (wealth_after − wealth_before) − φ · Δ_target²`
  - P&L is marked at true V*; risk penalty uses the *chosen* Δ_target (before execution)

**`step_environment!` signature:**
```julia
step_environment!(env, portfolio, pf::ParticleFilter, action::MarketMakingAction, config, rng)
```
Returns `(next_state, reward, done, StepInfo)`. Step order: quote → fill against V* → hedge → spot step → expiry check → particle filter update → reward.

**`FillOutcome`:** holds `bid_filled`, `ask_filled`, `bid_price`, `ask_price`, `V_market`, and `f_t::Int`.

## Hardy (2001) Regime Parameters

```julia
VolModel([0.121, 0.269], transition_matrix = [0.9982 0.0018; 0.0022 0.9978])
# Stationary: π₁ ≈ 0.55 (low vol), π₂ ≈ 0.45 (high vol)
# oracle_σ(env) = sqrt(Σⱼ pᵢⱼ σⱼ²)  ← used by analytical benchmarks
```

Constant-vol baseline: `VolModel([0.20])`.

## SimConfig Defaults

`A=140, k=6, Δt=1/252, T_option=30, n_options_per_episode=5, κ=0.001, φ=0.01, γ_market=0.1`.

## Analytical Benchmarks

Four policies in `7_benchmarks.jl`, each returning `MarketMakingAction(δ, Δ_target)`:
- **GLF-T+WW**: Guéant-Lehalle-Fernandez-Tapia spread + Whalley-Wilmott no-trade band
- **GLF-T+Naive**: GLF-T spread + always hedge to Δ_target=0
- **Naive+WW**: fixed $0.10 half-spread + WW band
- **Naive+Naive**: fixed $0.10 half-spread + always hedge to 0

All benchmarks receive `oracle_σ` (not σ_hat) so performance gaps vs RL reflect policy quality, not information disadvantage.

## Julia Notes

- Use `--project=.` to load `StatsBase`, `Distributions`, `Plots`, `StatsPlots`
- Top-level `for` loops in scripts need `let` blocks to avoid soft-scope `UndefVarError` when assigning variables
- `Float32` for particle log-σ values; `Float64` for log-weights (log-sum-exp normalization)
Core POMDP Solving & RL

Reminder of available packages:
- POMDPs.jl — Core POMDP interface and abstract types; everything else builds on this
- POMCPOW.jl — Partially Observable Monte Carlo Planning with Optimistic Widening (online solver; ideal for continuous belief spaces)
- ParticleFilters.jl — Core particle filter machinery; you'll use this heavily for tracking bt=P(σt∗∣FtMM)b_t = P(\sigma_t^* | \mathcal{F}_t^{MM})
bt​=P(σt∗​∣FtMM​)
- RobustAdaptiveMetropolisSampling.jl — For resampling strategies in particle filters
- StaticArrays.jl — Fixed-size arrays for state tuples (St,Vt∗,σt∗,qt,τt)(S_t, V_t^*, \sigma_t^*, q_t, \tau_t)
(St​,Vt∗​,σt∗​,qt​,τt​); performance-critical
Parameters.jl — Struct definition with defaults (@with_kw) for cleaner state/action/reward encoding
Plots.jl or Makie.jl — Plotting cumulative P&L, inventory paths, quote skewing dynamics over episodes
Statistics.jl — Mean/std for evaluating policy performance across episodes
Random.jl — Seeding and reproducibility for simulations

## Other Notes
Before editing a file provide a quick summary of the changes and why you are making them before proceeding. Explain your logic as you go.
