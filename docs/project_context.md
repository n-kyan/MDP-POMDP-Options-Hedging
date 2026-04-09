# Project Context: Options Market Making Under Uncertainty

## One-Line Summary

A market maker quotes bids and asks on options at a single strike, manages inventory by delta-hedging the underlying, and jointly optimizes spread width and hedge ratio using MDP/RL/POMDP solvers. The market has perfect regime knowledge; in Level 3, the agent must infer the market's pricing from returns and fill asymmetry.

---

## Project Purpose and Stakeholders

This is a joint final project for two CU Boulder courses:

- **ASEN/CSCI 5264 — Decision Making Under Uncertainty (DMU):** Taught by Professor Zachary Sunberg (primary author of POMDPs.jl). Paper due **April 30**, 4–8 pages, focused on algorithms and DMU course content.
- **Applied Derivatives:** Paper due **April 17**, 8–10 pages, focused on the financial modeling and derivatives theory.

**Recruiting goal:** This project is intended as a portfolio piece for quantitative trading firm applications. The project is implemented in Julia.

---

## Market Model: Perfect-Knowledge Market

The market (the aggregate of all participants) has **perfect knowledge** of the current volatility regime and transition probabilities. This is the most important modeling decision in the project.

### Why the market is omniscient

Real markets aggregate information from thousands of participants — order flow, options market dynamics, news, macro data. No single market maker has better fundamental pricing than the market consensus. The market maker's edge comes from providing liquidity and earning the spread, not from better directional bets.

### How V_market is computed

The market prices using a **transition-weighted Black-Scholes price**:

$$V_{\text{market}} = \sum_j P(\text{regime}_{t+1} = j \mid \text{regime}_t = i) \times V_{\text{BS}}(\sigma_j)$$

This accounts for the fact that even knowing the current regime, there is a small probability of transitioning next step. With Hardy (2001) daily transition probabilities >99.7% on the diagonal, this is very close to $V_{\text{BS}}(\sigma_{\text{true}})$ but correctly reflects that vol is not actually constant — it can switch.

In code: `compute_market_belief(vol_state)` returns the transition matrix row for the current regime, and `bs_all_belief_weighted` computes the weighted price.

### Information Hierarchy

| Entity | Knows | Computes V from |
|---|---|---|
| Market (= God = Simulator) | True regime + transition matrix | Transition-weighted BS (perfect) |
| Agent (Level 1) | σ is constant and known | V_BS(σ) directly |
| Agent (Level 2) | True regime + transition matrix | Same as market (fills are symmetric) |
| Agent (Level 3) | Returns + fill outcomes only | Hamilton filter belief-weighted BS (converging toward market) |

### What the agent learns from fills (Level 3)

The fill signal does NOT make the agent smarter than the market. It helps the agent **track where the market is pricing**. When the agent's V_believed diverges from V_market (because the agent's Hamilton filter hasn't caught up to a regime switch that the market already knows about), fills become asymmetric. This asymmetry tells the agent "your quotes are off-center — adjust toward where the market actually is."

The performance gap between Level 2 (agent matches market) and Level 3 (agent must infer market pricing) quantifies the **cost of partial observability**. The gap between a Level 3 agent with fill signal and one with returns only quantifies the **value of the fill signal** for tracking the market.

---

## Research Foundation

| Component | Paper | What We Take |
|---|---|---|
| Fill intensity model | Avellaneda & Stoikov (2008) | λ(δ) = Ae^{-kδ}, simulation framework |
| Spread benchmark (primary) | Guéant, Lehalle & Fernandez-Tapia (2013) Prop. 3 | Asymptotic bid/ask depths with bounded inventory |
| Regime-switching parameters | Hardy (2001) Table 1 | σ₁ = 12.1%, σ₂ = 26.9%, transition matrix |
| Regime-switching methodology | Hamilton (1989) | Markov-switching model, Hamilton filter |
| Hedge benchmark (simple) | Leland (1985) | Modified volatility σ̂ |
| Hedge benchmark (optimal) | Whalley & Wilmott (1997) | No-trade bandwidth H |
| Reward function structure | Cartea, Jaimungal & Penalva (2015) Ch. 10 | Running penalty, adapted from q² to Δ_net² |
| Fill asymmetry as signal | Tsaknaki, Lillo & Mazzarisi (2024) | Order flow carries regime information |

**Deliberate departures from AS:**
- No hard inventory limits (soft penalty via φ·Δ_net² instead)
- Market has perfect regime knowledge (AS assumes market is the reference price without specifying its information set)
- Options as asset class (AS models stocks)

---

## Environment Design

### Spot Price Dynamics

2-state regime-switching GBM with **risk-neutral drift** (μ = r in both regimes).

**Parameters from Hardy (2001), converted to daily:**

| Parameter | Regime 1 (Low Vol) | Regime 2 (High Vol) |
|---|---|---|
| σ (annualized) | 12.1% | 26.9% |

**Daily transition matrix:**

|  | → Regime 1 | → Regime 2 |
|--|---|---|
| From Regime 1 | 0.9982 | 0.0018 |
| From Regime 2 | 0.0022 | 0.9978 |

**Stationary distribution:** π₁ ≈ 0.55, π₂ ≈ 0.45.

### Sequential Option Lifetimes

Each option lives ~63 trading days. When expired, cash carries forward, new ATM option starts at K = round(S). Episodes consist of 4-8 sequential options (~1-2 years), ensuring 1-3 regime switches per episode.

### Fill Model

From AS (2008) eq. 2.11: λ(δ) = A·exp(-k·δ). Fill probabilities computed against V_market (the market's perfect-knowledge price). Agent quotes around V_believed. Asymmetric fills in Level 3 signal that the agent's pricing is off-center relative to the market.

No hard inventory limits. The reward penalty manages inventory risk.

### Spread Width Action Levels

| Level | Half-spread δ | Fill Rate (approx) |
|---|---|---|
| 1 | $0.05 | ~41% |
| 2 | $0.10 | ~30% |
| 3 | $0.20 | ~17% |
| 4 | $0.40 | ~5% |
| 5 | $0.80 | ~0.5% |

### Hedge Action Levels

{0%, 25%, 50%, 75%, 100%, 125%} of current net delta exposure.

### Combined Action Space

5 spread × 6 hedge = **30 discrete actions**.

---

## MDP / POMDP Formulation

### State Space

| Component | Description |
|---|---|
| Spot price S | Current underlying price |
| Time to expiry τ | Countdown to current option expiry |
| Call inventory q_calls | Signed count of call contracts |
| Spot inventory q_spot | Shares of underlying held for hedging |
| Cash | Cumulative cash balance |
| Regime belief | Agent's belief over regimes (varies by level) |

### Reward Function

r_t = ΔPnL_t - φ · Δ_net²

Four P&L components: mark-to-market on options, spread capture from fills, hedge P&L on underlying shares, hedge transaction costs.

---

## Three Levels of Complexity

### Level 1 — Constant Volatility (Baseline)
Fully observable MDP, single regime. V_believed = V_market = V_BS(σ). Fills are symmetric. Agent optimizes spread and hedge. Solved via value iteration. Benchmarked against AS, GLF-T, Leland, Whalley-Wilmott.

### Level 2 — Known Regime (Main Approach)
Fully observable MDP with regime-switching. Agent has same info as market (perfect regime knowledge). V_believed = V_market. Fills are symmetric. The value comes from adapting spread/hedge to regime changes. Solved via MCTS and DQN.

### Level 3 — Hidden Regime (POMDP Stretch Goal)
Agent does NOT know the true regime. Must infer V_market from returns and fill patterns via Hamilton filter. V_believed may lag behind V_market after regime switches, creating asymmetric fills. The agent's task is to track the market, not beat it. Solved via QMDP and POMCPOW.

**Key measurements:**
- Cost of partial observability: Level 2 minus Level 3 performance
- Value of fill signal: Level 3 (returns + fills) minus Level 3 (returns only)

---

## Solver Stack

| Solver | Type | Level | From Scratch? |
|---|---|---|---|
| Value Iteration | Exact MDP | L1 | Yes |
| MCTS (DPWSolver) | Online MDP | L2 | POMDPs.jl |
| DQN | Model-free RL | L2 | Yes (Flux.jl) |
| Hamilton Filter | Belief updater | L3 | Yes |
| QMDP | POMDP approx | L3 | Yes |
| POMCPOW | Online POMDP | L3 | POMDPs.jl |

---

## Timeline

- **April 14:** Core environment complete, Level 1 value iteration running, baseline results
- **April 17:** Derivatives paper submitted (finance focus, 8–10 pages)
- **April 24:** Level 2 (MCTS + DQN) and Level 3 (QMDP + POMCPOW) complete
- **April 30:** DMU paper submitted (algorithms focus, 4–8 pages)
