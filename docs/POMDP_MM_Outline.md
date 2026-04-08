# Final Project Proposal
**ASEN/CSCI 5264 — Decision Making Under Uncertainty**
Kyan | April 2026

---

## Options Market Making Under Uncertainty: A POMDP Approach to Joint Spread and Hedge Optimization

---

## Problem Statement

An options market maker simultaneously quotes bids and asks on calls and puts at a single strike and manages the resulting inventory by delta-hedging in the underlying. At each timestep, the agent makes two coupled decisions: (1) *spread width* — how wide to quote around its believed fair value, and (2) *hedge target* — what fraction of its net delta exposure to offset via trades in the underlying.

The core tension is that wider spreads earn more edge per fill but attract less order flow, while frequent hedging reduces directional risk but incurs transaction costs. The environment is stochastic (GBM or regime-switching spot dynamics), and at the highest level, the true volatility regime is *hidden* — making this a POMDP. The agent receives observations of spot price returns and fill asymmetry (lopsided fills signal mispricing and are informative about the hidden regime).

**Formal structure:**

- **State:** (net inventory, net portfolio delta $\Delta$, portfolio gamma $\Gamma$, moneyness $S/K$, time to expiry $\tau$, current hedge position, volatility regime)
- **Action:** (spread width, hedge target) — discrete cross product, ~24–30 actions
- **Observation (POMDP):** state minus true regime, plus log-return and fill outcome
- **Reward:** mark-to-market P&L + spread capture $-$ hedging transaction costs $-$ $\lambda \cdot \text{P\&L}^2$
- **Transition:** GBM (L1), 3-state regime-switching Markov chain (L2/L3)

---

## Level 1 — Minimum Working Example

*Fully observable MDP with constant volatility (pure GBM dynamics)*

The agent knows the true volatility at all times. The state space is discretized (~2,000 states over $\Delta$, $m$, $\tau$, hedge position). The agent learns an optimal joint (spread, hedge) policy via **value iteration**, implemented from scratch.

Benchmarks: (1) naive full-delta hedge every period (Black-Scholes baseline), (2) Leland's modified-volatility heuristic, (3) Whalley-Wilmott no-trade bandwidth.

- **From scratch:** full environment (Black-Scholes pricing, GBM simulation, fill model, P&L accounting), value iteration solver.
- **Off-the-shelf:** none at this level.

---

## Level 2 — Main Approach

*Fully observable MDP with regime-switching volatility*

Volatility follows a 3-state Markov chain — low ($\sigma \approx 10\%$), medium ($\sigma \approx 20\%$), and high ($\sigma \approx 40\%$) — calibrated to S&P 500 daily returns (Cerboni Baiardi et al., 2020), with stationary distribution approximately 60% / 33% / 7% and self-transition probabilities exceeding 0.96. The true regime is included in the state. The agent must learn to adapt spread and hedge behavior across all three regimes. Two solvers are applied and compared: **MCTS with double progressive widening** (DPWSolver from POMDPs.jl) and **Deep Q-Network** (DQN, implemented from scratch using Flux.jl). DQN encodes the regime as a feature and learns a nonlinear value function over the expanded state space.

- **From scratch:** DQN training loop, replay buffer, $\varepsilon$-greedy exploration, neural network architecture (Flux.jl).
- **Off-the-shelf:** DPWSolver (POMDPs.jl), Flux.jl for autodiff/neural network primitives.

---

## Level 3 — Stretch Goal

*POMDP with hidden volatility regime*

The true volatility regime is now unobserved. The agent maintains a belief distribution over three regimes, updated each step via a Bayesian filter using two signals: (1) the magnitude of log price returns (higher vol $\Rightarrow$ larger moves) and (2) fill asymmetry (if the agent underestimates vol, its ask is underpriced $\Rightarrow$ more ask fills than bid fills). Pricing uses the belief-weighted Black-Scholes price (not BS at mean vol, due to Jensen's inequality). The belief state lives on a 2D simplex — two free probabilities since the three must sum to one.

Solvers applied: **QMDP** (fast approximation, implemented from scratch), **POMCPOW** (online tree search under partial observability, from POMDPs.jl). Key deliverable: quantify the *cost of partial observability* — the performance gap between L2 (regime known) and L3 (regime hidden).

- **From scratch:** Bayesian belief updater, QMDP solver, POMDPs.jl-compatible `gen()` interface.
- **Off-the-shelf:** POMCPOW (POMDPs.jl).

---

## Note on Scope Change

An earlier version of this proposal framed the project as a single-option delta-hedging MDP. Following deeper engagement with the problem, the scope evolved to include market-making (joint spread and hedge optimization) and a POMDP layer modeling hidden volatility regimes — both more realistic and richer in DMU course content. The core infrastructure (Black-Scholes environment, discretized MDP, value iteration) is unchanged from the original framing; the expansion adds the fill model, regime dynamics, and POMDP belief update as new components.
