# Options Market-Making POMDP

A Julia implementation of a POMDP-based market maker that quotes bids and asks on a single-strike ATM call option, delta-hedges the underlying, and jointly optimizes spread width and hedge ratio under volatility regime uncertainty.

## Problem Formulation

The agent acts as a market maker in continuous time (discretized to daily steps, `Δt = 1/252`). At each step it chooses:

- **Half-spread** `δ ∈ [κ·S·|Δ̂|, ]` — how wide to quote around its believed fair value
- **Target portfolio delta** `Δ_target ∈ [min(0, Δ̂_P), max(0, Δ̂_P)]` — how aggressively to hedge

The true volatility follows a **Hardy (2001) two-regime Markov chain** (`σ₁ = 12.1%`, `σ₂ = 26.9%`), which the agent does not observe. The market prices options using the true regime's forward-looking vol; the agent infers vol from a particle filter over log-returns and fill outcomes.

**Reward:** `r_t = ΔWealth_t − φ · Δ_target²` (mark-to-market P&L minus a delta-penalty)

## Repository Structure

```
src/
  1_types.jl           — All shared structs (SimConfig, VolModel, MarketMakingAction, AgentState, ...)
  2_black_scholes.jl   — BS pricing and Greeks (price, Δ, Γ, Vega, Theta)
  3_spot_dynamics.jl   — GBM spot evolution with Markov regime switching
  4_fills.jl           — Avellaneda-Stoikov fill model; quote computation
  5_portfolio.jl       — Portfolio accounting, reward, and hedge execution
  6_environment.jl     — Episode runner: initialize_episode!, step_environment!
  7_benchmarks.jl      — Analytical policies: GLF-T+WW, GLF-T+Naive, Naive+WW, Naive+Naive
  8_evaluation.jl      — Monte Carlo evaluation and figure generation
  9_value_iteration.jl — Value iteration on constant-vol MDP (oracle baseline)
  9_mcts_mdp.jl        — MCTS on oracle MDP (regime observable)
  10_pomcpow.jl        — POMCPOW solver with GLFT-informed action widening
  10_pomdp_interface.jl — Shared POMDPs.jl type definitions
  11_dqn.jl            — Deep Q-network solver
  12_belief_updater.jl — Particle filter for σ* inference
  13_qmdp.jl           — QMDP approximation

tests/
  test_1_2_3.jl        — Types, Black-Scholes, spot dynamics (42 tests)
  test_4.jl            — Fill model (29 tests)

results/
  fig1_pnl_distributions.png
  fig2_spread_vs_tau.png
  fig3_hedge_behavior.png
  fig4a_cumulative_pnl_const.png
  fig4b_cumulative_pnl_hardy.png
  table1_summary.txt

docs/
  DMU Final Paper.md   — Single source of truth for all formulations
  implementation_plan.md
  paper_outline.md
  readings/            — Reference papers (Hardy 2001, GLF-T 2013, Whalley-Wilmott, etc.)
```

## Module Load Order

Modules use `include()` and must be loaded in dependency order. Module 12 (particle filter) loads before Module 6 (environment) because the environment calls `update!` on the particle filter.

```
1 → 2 → 3 → 4 → 5 → 12 → 6   (core environment stack)
7 includes the above stack      (benchmarks)
8 includes 7                    (evaluation)
10 includes 7                   (POMCPOW)
```

## Running

```bash
# Always use --project=. so Julia finds StatsBase and other dependencies

# Run unit tests
julia --project=. tests/test_1_2_3.jl   # 42 tests: types, Black-Scholes, spot dynamics
julia --project=. tests/test_4.jl        # 29 tests: fill model

# Run the full benchmark evaluation (produces figures + table in results/)
julia --project=. src/8_evaluation.jl

# Run POMCPOW diagnostic trace (5-episode step-by-step output)
# From within 10_pomcpow.jl, call:
#   diagnose_pomcpow(VM_HARDY, SIM_CONFIG; n_episodes=5, seed=42)

# Run POMCPOW evaluation (30 episodes)
#   evaluate_pomcpow(VM_HARDY, SIM_CONFIG, 30, 42)
```

## Key Abstractions

### Action Space (Continuous)

```julia
struct MarketMakingAction
    δ::Float64        # half-spread in dollars
    Δ_target::Float64 # desired net portfolio delta after hedging
end
```

No discrete grid — solvers output these two floats directly.

### Belief (Particle Filter over σ\*)

```julia
struct ParticleFilter
    log_σ::Vector{Float32}  # log of each particle's σ estimate
    log_w::Vector{Float64}  # log unnormalized weights
    n::Int
end
```

- Prior: log-normal centered at `log(0.20)`, std `0.5` in log-space
- `update!(pf, r_t, config)`: weights via GBM log-likelihood; systematic resample when ESS < n/2
- `get_σ_hat(pf)` → weighted-mean σ estimate

### Observation

```julia
struct AgentState
    r_t::Float64      # log-return this step
    hat_Δ_P::Float64  # net portfolio delta
    hat_Γ_P::Float64  # portfolio gamma
    f_t::Int          # fill signal: +1 bid, -1 ask, 0 otherwise
    τ::Float64        # time to expiry (years)
    σ_hat::Float64    # particle-filter vol estimate
    hat_V::Float64    # BS price under σ_hat
    hat_Δ::Float64    # per-contract delta under σ_hat
    S::Float64        # spot price
end
```

### Volatility Environment

```julia
# Hardy (2001) two-regime model
VM_HARDY = VolModel(
    [0.121, 0.269],
    transition_matrix = [0.9982 0.0018;
                         0.0022 0.9978]
)
# Stationary distribution: π₁ ≈ 55% low-vol, π₂ ≈ 45% high-vol

# Constant-vol baseline
VM_CONST = VolModel([0.20])
```

## Analytical Benchmark Policies

All benchmarks receive `oracle_σ` (transition-row-weighted true vol), ensuring performance gaps vs RL reflect policy quality rather than information asymmetry.

| Policy | Spread | Hedge |
|--------|--------|-------|
| **GLF-T + WW** | Guéant-Lehalle-Fernandez-Tapia optimal spread | Whalley-Wilmott no-trade band |
| **GLF-T + Naive** | GLF-T spread | Always hedge to Δ = 0 |
| **Naive + WW** | Fixed $0.10 half-spread | Whalley-Wilmott no-trade band |
| **Naive + Naive** | Fixed $0.10 half-spread | Always hedge to Δ = 0 |

### GLF-T Half-Spread Formula

`δ* = γ·|Γ|·S²·σ²·τ + (2/γ)·ln(1 + γ/k)`

### Whalley-Wilmott No-Trade Band Halfwidth

`H = (3κ/(2φ) · Γ²·S²·σ²)^(1/3) · Δt^(1/3)`

## Benchmark Results (1000 episodes each)

### Constant Volatility (σ = 0.20)

| Policy | Mean P&L | Std P&L | Sharpe |
|--------|----------|---------|--------|
| GLF-T + WW | 3.71 | 2.22 | **1.67** |
| GLF-T + Naive | 3.18 | 2.34 | 1.36 |
| Naive + WW | 5.13 | 3.73 | 1.38 |
| Naive + Naive | 4.55 | 3.58 | 1.27 |

### Regime-Switching (Hardy 2001)

| Policy | Mean P&L | Std P&L | Sharpe |
|--------|----------|---------|--------|
| GLF-T + WW | 4.06 | 2.04 | **1.99** |
| GLF-T + Naive | 3.53 | 2.49 | 1.42 |
| Naive + WW | 3.99 | 3.87 | 1.03 |
| Naive + Naive | 3.69 | 3.32 | 1.11 |

GLF-T + WW dominates on Sharpe in both environments. The WW hedge band significantly reduces variance relative to always-hedging, while the GLF-T spread captures the inventory risk premium that a flat $0.10 spread misses.

## POMCPOW Solver

`10_pomcpow.jl` implements a POMCPOW planner that treats vol as a continuous latent variable (no Hardy knowledge). Key design choices:

- **Agent's world model**: log-vol random walk (`log σ_{t+1} = log σ_t + N(0, ξ²·Δt)`) rather than Hardy regime switching
- **Action widening**: samples `δ` from a log-normal centered at the GLFT value with `σ=0.3` in log-space — keeps GLFT as the prior while allowing ≈2× exploration range
- **Rollout policy**: GLF-T + WW using each particle's `σ_particle`
- **Rao-Blackwellization**: fully-observable quantities (S, q, τ, cash) are synced from the true environment after each belief update; only `σ_particle` remains uncertain

```julia
# Diagnostic trace (5 episodes, step-by-step δ comparison)
diagnose_pomcpow(VM_HARDY, SIM_CONFIG; n_episodes=5, seed=42)

# Full evaluation
evaluate_pomcpow(VM_HARDY, SIM_CONFIG, 30, 42)
```

## SimConfig Defaults

| Parameter | Value | Description |
|-----------|-------|-------------|
| `S0` | 100.0 | Initial spot price |
| `r` | 0.05 | Risk-free rate (annualized) |
| `Δt` | 1/252 | Timestep (1 trading day) |
| `T_option` | 30 | Trading days per option lifetime |
| `n_options_per_episode` | 5 | Options per episode (~150 trading days) |
| `κ` | 0.001 | Proportional hedge transaction cost (10 bps) |
| `A` | 140.0 | Fill intensity scale (Avellaneda-Stoikov) |
| `k` | 6.0 | Fill decay rate |
| `γ_market` | 0.1 | Risk aversion in GLF-T formula |
| `φ` | 0.01 | Delta penalty weight in reward |

## Dependencies

Managed via `Project.toml`. Key packages:

- `POMDPs.jl`, `POMCPOW.jl`, `POMDPTools.jl`, `BasicPOMCP.jl` — POMDP solver infrastructure
- `MCTS.jl` — Monte Carlo tree search for oracle MDP
- `ParticleFilters.jl` — Particle filter belief updater
- `Distributions.jl` — Probability distributions
- `Flux.jl` — Neural networks (DQN)
- `Plots.jl`, `StatsPlots.jl` — Figure generation
- `StatsBase.jl` — Weighted sampling

```bash
# Install all dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Julia Notes

- Always run with `--project=.` or Julia will not find the registered packages
- Top-level `for` loops in scripts require `let` blocks to avoid soft-scope `UndefVarError`
- `Float32` for particle log-σ values; `Float64` for log-weights (log-sum-exp normalization)
