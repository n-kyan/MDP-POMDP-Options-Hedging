# Project Context: Options Market Making Under Uncertainty

## One-Line Summary

A market maker quotes bids and asks on options at a single strike, manages inventory by delta-hedging the underlying, and jointly optimizes spread width and hedge ratio. Modules 1–8 are complete. The derivatives paper (due April 17) uses analytical benchmark results from Module 8. The DMU paper (due April 30) adds RL solvers (Modules 9–13).

---

## Project Purpose and Stakeholders

Joint final project for two CU Boulder courses:

- **CSCI 5264 — Decision Making Under Uncertainty (DMU):** Taught by Professor Zachary Sunberg (primary author of POMDPs.jl). Paper due **April 30**, 4–8 pages, focused on algorithms and DMU course content.
- **Applied Derivatives:** Taught by Professor Daniel Brown. Paper due **April 17**, 8–10 pages, focused on financial modeling and derivatives theory.

**Recruiting goal:** Portfolio piece for quantitative trading firm applications. Implemented in Julia.

---

## Current Implementation Status (as of April 13, 2026)

### Completed Modules

| Module | File | Status |
|---|---|---|
| 1 | `1_types.jl` | Complete |
| 2 | `2_black_scholes.jl` | Complete |
| 3 | `3_spot_dynamics.jl` | Complete |
| 4 | `4_fills.jl` | Complete |
| 5 | `5_portfolio.jl` | Complete |
| 6 | `6_environment.jl` | Complete — returns `(next_state, reward, done, StepInfo)` |
| 7 | `7_benchmarks.jl` | Complete |
| 8 | `8_evaluation.jl` | Complete — runs 4 policies, 2 environments, produces all figures |

### Pending Modules (for DMU paper, April 18–30)

| Module | File | Target |
|---|---|---|
| 9 | `9_value_iteration.jl` | April 18 |
| 10 | `10_pomdp_interface.jl` | April 19 |
| 11 | `11_dqn.jl` | April 20 |
| 12 | `12_belief_updater.jl` | April 22 |
| 13 | `13_qmdp.jl` | April 23 |

---

## Derivatives Paper Scope (April 17 deadline)

The derivatives paper does **not** include RL results. It covers:
- The simulation framework (Modules 1–6)
- The four analytical benchmark policies (Module 7)
- The Monte Carlo evaluation results (Module 8)

The RL solver comparison is explicitly deferred to the DMU paper and framed as "future work" in the derivatives paper.

---

## Benchmark Results Summary (1,000 episodes, seed=42)

### Four Policies Evaluated

| Policy | Spread Component | Hedge Component |
|---|---|---|
| GLF-T + WW | Guéant-Lehalle-Fernandez-Tapia optimal | Whalley-Wilmott no-trade band |
| GLF-T + Naive | Guéant-Lehalle-Fernandez-Tapia optimal | Always target net_Δ = 0 |
| Naive + WW | Fixed $0.10 half-spread | Whalley-Wilmott no-trade band |
| Naive + Naive | Fixed $0.10 half-spread | Always target net_Δ = 0 |

**Oracle σ:** All policies receive the transition-row-weighted variance-equivalent σ = sqrt(Σⱼ pᵢⱼ σⱼ²), matching what the market uses via `perfect_regime_belief`. This is the correct σ input for GLF-T and WW in a two-regime world.

### Key Results

**Constant Volatility (σ = 0.20):**

| Policy | Mean P&L | Std P&L | Sharpe | Spread($) | Hedge% | \|net Δ\| | Hedge Cost |
|---|---|---|---|---|---|---|---|
| GLF-T + WW | 10.65 | 6.06 | 1.76 | 0.53 | 45.1% | 0.083 | 3.73 |
| GLF-T + Naive | 10.09 | 4.18 | **2.41** | 0.53 | 100.0% | 0.054 | 4.77 |
| Naive + WW | **14.48** | 11.93 | 1.21 | 0.10 | 62.5% | 0.220 | 14.45 |
| Naive + Naive | 11.39 | 10.08 | 1.13 | 0.10 | 100.0% | 0.138 | 18.02 |

**Regime-Switching (Hardy 2001):**

| Policy | Mean P&L | Std P&L | Sharpe | Spread($) | Hedge% | \|net Δ\| | Hedge Cost |
|---|---|---|---|---|---|---|---|
| GLF-T + WW | 11.57 | 6.46 | **1.79** | 0.51 | 44.6% | 0.087 | 4.05 |
| GLF-T + Naive | 10.70 | 4.97 | 2.15 | 0.51 | 100.0% | 0.057 | 5.15 |
| Naive + WW | **13.69** | 14.35 | 0.93 | 0.10 | 62.2% | 0.225 | 14.88 |
| Naive + Naive | 10.89 | 11.86 | 0.92 | 0.10 | 100.0% | 0.139 | 18.12 |

### Three Core Findings from Results

**Finding 1 — Spread formula dominates variance reduction.**
GLF-T cuts Std P&L roughly in half versus Naive spread (6.06 vs 10–12). The hedging choice barely affects variance when spread is already wide. GLF-T+WW and GLF-T+Naive have nearly identical variance despite very different hedging behavior.

**Finding 2 — WW's value is cost savings, not delta management.**
GLF-T+WW saves ~$1/episode in hedge costs vs GLF-T+Naive with nearly the same P&L. For Naive spreads, WW saves ~$3.57/episode. Counterintuitively, WW actually produces *higher* |net Δ| than Naive hedge when paired with a tight spread — tight spreads generate frequent fills and rapid inventory accumulation; WW's no-trade band allows delta to grow while Naive constantly resets it. This is correct WW behavior (avoiding costly transactions) but reveals an important interaction: WW is not an unconditional improvement over Naive hedge.

**Finding 3 — Regime switching amplifies Naive policy variance but barely affects GLF-T.**
Std P&L for GLF-T+WW increases 6.6% under regime switching (6.06 → 6.46). For Naive+WW it increases 20.3% (11.93 → 14.35). This disproportionate sensitivity motivates the DMU paper's RL extension: policies that cannot adapt spread width to the current volatility regime pay a growing variance penalty as regime uncertainty increases.

---

## Market Model: Perfect-Knowledge Market

The market computes its consensus option price as a **transition-weighted Black-Scholes price**:

$$V_{\text{market}} = \sum_j P(\text{regime}_{t+1} = j \mid \text{regime}_t = i) \times V_{\text{BS}}(\sigma_j)$$

In code: `perfect_regime_belief(vs)` returns the transition matrix row for the current regime, and `bs_all_belief_weighted` computes the weighted price. The same belief vector used in the market's pricing is also used to compute the oracle σ passed to analytical benchmark policies.

### Information Hierarchy

| Entity | Knows | Computes V from |
|---|---|---|
| Market (= God = Simulator) | True regime + transition matrix | Transition-weighted BS (perfect) |
| Agent (Level 1) | σ is constant and known | V_BS(σ) directly |
| Agent (Level 2) | True regime + transition matrix | Same as market (fills are symmetric) |
| Agent (Level 3) | Returns + fill outcomes only | Hamilton filter belief-weighted BS |
| Benchmark policies (Modules 7–8) | Oracle σ (transition-weighted variance-equivalent) | GLF-T / WW formulas with oracle σ |

---

## Environment Design

### Spot Price Dynamics

2-state regime-switching GBM with risk-neutral drift (μ = r).

**Hardy (2001) parameters:**

| Parameter | Regime 1 (Low Vol) | Regime 2 (High Vol) |
|---|---|---|
| σ (annualized) | 12.1% | 26.9% |
| Stationary weight | π₁ ≈ 0.55 | π₂ ≈ 0.45 |

**Daily transition matrix:**

|  | → Regime 1 | → Regime 2 |
|--|---|---|
| From Regime 1 | 0.9982 | 0.0018 |
| From Regime 2 | 0.0022 | 0.9978 |

Average regime duration: ~556 days (regime 1), ~455 days (regime 2). Very persistent — regime switches are rare events within a single episode.

**Constant vol baseline:** σ = 0.20, computed as the variance-weighted stationary average sqrt(π₁σ₁² + π₂σ₂²) ≈ 0.20.

### SimConfig Defaults

```julia
S0 = 100.0, r = 0.05, Δt = 1/252
T_option = 63 days, n_options_per_episode = 8  # 504 steps per episode
κ = 0.001 (10 bps transaction cost)
A = 140.0, k = 6.0 (AS fill model)
γ_market = 0.1, φ = 0.01
spread_levels = [0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
Δ_targets = [:no_trade, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0,
              0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
```

### Action Space

6 spread levels × 14 hedge targets = **84 discrete actions**.

### StepInfo Struct

`step_environment!` returns `(next_state, reward, done, StepInfo)` where:

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

---

## Key Design Decisions (Resolved)

**Oracle σ for benchmarks:** Transition-row-weighted variance-equivalent σ = sqrt(Σⱼ pᵢⱼ σⱼ²), not current-regime σ. Gives policies the same information the market uses — any performance difference vs the market is purely policy quality, not information asymmetry. Consistent with Level 2 RL where agent also has perfect regime knowledge.

**Reward function:** r = pnl - φ·Δ_net² (convex delta penalty, not |Δ|). Differentiable, proportional to P&L variance.

**WW band target:** Trade to band edge (sign(net_Δ) × H), not to zero. Consistent with WW's derivation — the optimal policy outside the band is to trade back to the nearest boundary, not to fully neutralize.

**Belief-weighted pricing:** Must average BS prices across regimes with regime weights, not BS at mean σ. Jensen's inequality: E[V(σ)] ≥ V(E[σ]) for convex BS formula.

**Theta:** Computed analytically in `bs_all`, not as finite difference in portfolio.jl.

**No hard inventory limits:** Soft penalty via φ·Δ_net² instead. Avoids AS's bounded-inventory constraint, which is required for closed-form solution but not for RL.

---

## MDP / POMDP Formulation

### State Space (AgentState)

S, τ, net_Δ, net_Γ, net_ν, net_Θ, regime_belief

### Reward

r_t = ΔPnL_t - φ · Δ_net²

Four P&L components: mark-to-market, spread capture, hedge P&L, hedge transaction cost.

---

## Three Levels of Complexity

### Level 1 — Constant Volatility
Single regime, no switching. V_believed = V_market = V_BS(σ). Fills symmetric. Solved via value iteration (Module 9, target April 18).

### Level 2 — Known Regime
Agent has same info as market. V_believed = V_market. Fills symmetric. Solved via MCTS (Module 10) and DQN (Module 11).

### Level 3 — Hidden Regime (POMDP)
Agent infers regime from returns + fill asymmetry via Hamilton filter (Module 12). Solved via QMDP (Module 13) and POMCPOW (Module 10).

---

## Solver Stack (Target: April 18–30)

| Solver | Type | Level | Module |
|---|---|---|---|
| Value Iteration | Exact MDP | L1 | 9 |
| MCTS (DPWSolver) | Online MDP | L2 | 10 |
| DQN | Model-free RL | L2 | 11 |
| Hamilton Filter | Belief updater | L3 | 12 |
| QMDP | POMDP approx | L3 | 13 |
| POMCPOW | Online POMDP | L3 | 10 |

---

## Timeline

- **April 13:** Modules 1–8 complete, all benchmark results in hand ✓
- **April 15–17:** Write and submit derivatives paper
- **April 18–24:** Implement Modules 9–13 (RL solvers)
- **April 25–30:** Write and submit DMU paper
