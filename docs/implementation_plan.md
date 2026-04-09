# Implementation Plan: Options Market Making Under Uncertainty

## Overview

This plan organizes the implementation into modules with explicit dependencies, build order, and deadline targets. Each module specifies what is built from scratch versus what uses existing libraries, and which research paper provides the foundation.

The two deadlines drive the priority ordering. April 17 (derivatives paper) requires a working Level 1 environment, value iteration solver, all analytical benchmarks, and baseline results. April 30 (DMU paper) adds Level 2 (MCTS + DQN) and Level 3 (QMDP + POMCPOW), plus the cost-of-partial-observability analysis.

---

## Market Model: Perfect-Knowledge Market

The market (the aggregate of all participants) has **perfect knowledge** of the current volatility regime and transition probabilities. This reflects the reality that the aggregate market incorporates far more information than any single market maker and is the best available estimator of fair value.

The market computes its consensus option price as:

$$V_{\text{market}} = \sum_j P(\text{regime}_{t+1} = j \mid \text{regime}_t = i) \times V_{\text{BS}}(\sigma_j)$$

where $i$ is the current (true) regime. This is the transition-row-weighted BS price — the correct one-step-ahead expected value given perfect regime knowledge and awareness of possible transitions. In practice, with daily transition probabilities >99.7% on the diagonal, this is very close to $V_{\text{BS}}(\sigma_{\text{true}})$ but is more intellectually defensible.

The **market and "God" (the simulator) are the same entity.** No separate market belief filter is needed. `environment.jl` computes $V_{\text{market}}$ directly from `vol_state.regime_idx` and the transition matrix via `compute_market_belief(vol_state)`.

The agent's goal is **not** to outsmart the market about fair value — it is to track where the market is pricing, earn the bid-ask spread, and manage the risk of accumulated inventory through hedging. The agent makes money from providing liquidity, not from making better directional bets than the market.

### Information Hierarchy

| Entity | Knows | Computes V from |
|---|---|---|
| Market (= God) | True regime + transition matrix | Transition-weighted BS (perfect) |
| Agent (Level 1) | σ is constant and known | V_BS(σ) directly (no uncertainty) |
| Agent (Level 2) | True regime + transition matrix | Same as market (symmetric fills, pure spread/hedge optimization) |
| Agent (Level 3) | Returns + fill outcomes only | Hamilton filter belief-weighted BS (imperfect, converging toward market) |

### Inventory Management

We remove the hard inventory bounds (max Q per side) from the original AS formulation. Instead, inventory risk is managed through the soft penalty $\phi \cdot \Delta_{\text{net}}^2$ in the reward function. This is more flexible than hard limits: the agent learns its own risk tolerance rather than having it imposed. AS required bounded inventory for their closed-form analytical solution; since we're solving with RL, we don't need that constraint. This is noted as a deliberate departure from AS in the paper.

---

## Module Inventory

### Module 1: Types and Configuration — `types.jl`

**From scratch.** Defines all shared data structures: `OptionContract` (strike K, option type), `HedgingState` (spot S, tau, inventory q_calls/q_puts, hedge shares q_spot, cash, regime_belief), `EnvironmentState` (wraps HedgingState with vol_state, current_options, options_completed), `MarketMakingAction` (spread_idx 1-5, hedge_idx 1-6), `SimConfig` (all simulation parameters), and `compute_market_belief` (returns the transition matrix row for the current regime).

The separation between `HedgingState` (the agent's view) and `EnvironmentState` (the full simulation state) enforces the information barrier. The agent never sees `vol_state` directly in Level 3; it only experiences the market's pricing through fill outcomes.

No `market_belief` field in EnvironmentState — the market has perfect knowledge, so its "belief" is computed on the fly from the true regime via `compute_market_belief(vol_state)`.

**Dependencies:** None. **Target:** April 9.

---

### Module 2: Black-Scholes Pricing and Greeks — `black_scholes.jl`

**From scratch.** Pure math functions: `bs_price`, `bs_Δ_Γ`, `bs_ν`, `bs_all` (returning a NamedTuple with price, Δ, Γ, ν), internal helper `_d1_d2`. Guard clause returns intrinsic value and limit Greeks when τ ≤ 0.

Belief-weighted pricing: `bs_all_belief_weighted(S, K, τ, σ_regimes, beliefs, r; call)` computes Σⱼ beliefs[j] × V_BS(σⱼ). Used for both V_market (beliefs = transition row from true regime) and V_believed (beliefs = agent's Hamilton filter output in Level 3).

**Dependencies:** None (pure math). **Target:** April 9.

---

### Module 3: Spot Price Simulation — `spot_dynamics.jl`

**From scratch.** Single `step(S, vs, config, rng)` function that transitions the regime (Markov chain) and advances the spot price (GBM with risk-neutral drift). Returns `(S_new, log_return)`.

**Dependencies:** types.jl. **Target:** April 9.

---

### Module 4: Fill Model — `fills.jl`

**From scratch.** Implements AS (2008) eq. 2.11.

Functions:
- `compute_quotes(V_believed, half_spread)` → `(; bid_price, ask_price)`
- `fill_probability(δ, A, k, Δt)` → Float64 (core AS formula)
- `simulate_fills(bid_price, ask_price, V_market, config, rng)` → `FillOutcome`
- `fill_probability_for_regime(bid_price, ask_price, V_market_j, config)` → `(; p_bid, p_ask)`
- `fill_outcome_likelihood(outcome, V_market_j, config)` → Float64

`FillOutcome` struct stores bid_filled, ask_filled, bid_price, ask_price, V_market — a self-contained record for testing, visualization, and the belief updater.

Fill probabilities are computed against V_market (the market's perfect-knowledge price). The agent quotes around V_believed. When V_believed ≠ V_market (Level 3 only), fills become asymmetric — signaling to the agent that its pricing is off-center relative to the market.

`fills.jl` is pure math — no inventory logic, no state mutation. `environment.jl` passes V_market and V_believed in.

**Dependencies:** types.jl (for SimConfig, FillOutcome). **Target:** April 10.

---

### Module 5: Portfolio and P&L Accounting — `portfolio.jl`

**From scratch.** Tracks inventory, aggregate Greeks, hedge shares, and cash. Computes per-step P&L from four sources: mark-to-market, spread capture, hedge P&L, and hedge transaction cost.

Key functions: `update_inventory!`, `execute_hedge!`, `compute_pnl`, `compute_reward` (implementing r = pnl - φ·Δ_net²), and `reset_for_new_option!`.

No hard inventory limits. The reward penalty φ·Δ_net² provides soft inventory management.

**Dependencies:** types.jl, black_scholes.jl. **Target:** April 11.

---

### Module 6: Environment Step Function — `environment.jl`

**From scratch.** Wires modules 2–5 into a single `step!` function:

1. Compute V_believed from agent's `regime_belief` via `bs_all_belief_weighted`
2. Compute quotes via `compute_quotes(V_believed, half_spread)`
3. Compute V_market: `compute_market_belief(vol_state)` → `bs_all_belief_weighted`
4. Execute hedge (trade shares to target, charge κ·|trade|·S)
5. Simulate fills against V_market via `simulate_fills`
6. Update inventory and cash
7. Step spot price and regime via `step(S, vs, config, rng)`
8. Recompute Greeks at new price
9. Compute P&L and reward (r = pnl - φ·Δ_net²)
10. Decrement τ; if expired, settle and start new option

In Levels 1-2, V_believed = V_market (agent has same info as market), so fills are symmetric. In Level 3, V_believed may lag behind V_market after regime switches, creating asymmetric fills that help the agent track the market.

**Dependencies:** All modules 1–5. **Target:** April 12.

---

### Module 7: Analytical Benchmarks — `benchmarks.jl`

**From scratch.** Spread benchmarks: AS optimal spread, GLF-T asymptotic depths. Hedge benchmarks: Leland modified delta, Whalley-Wilmott bandwidth. Includes `run_benchmark` for simulation comparison.

**Dependencies:** types.jl, black_scholes.jl, environment.jl. **Target:** April 13.

---

### Module 8: Evaluation and Visualization — `evaluation.jl`

**From scratch.** Monte Carlo evaluation, policy comparison, P&L distributions, policy surfaces, metrics computation.

**Dependencies:** types.jl, environment.jl, benchmarks.jl. Uses Plots.jl. **Target:** April 14.

---

### Module 9: Value Iteration Solver — `value_iteration.jl`

**From scratch.** Tabular VI for Level 1. ~2,000 states across (Δ_net, moneyness, τ, hedge_position). 30 actions.

**Dependencies:** types.jl, environment.jl. **Target:** April 13.

---

### Module 10: POMDPs.jl Interface — `pomdp_interface.jl`

**From scratch** (interface) + **POMDPs.jl** (solvers). Enables DPWSolver (MCTS) and POMCPOW.

**Dependencies:** types.jl, environment.jl. **Target:** April 18.

---

### Module 11: DQN Solver — `dqn.jl`

**From scratch** using Flux.jl. 3-layer MLP, replay buffer, ε-greedy, target network.

**Dependencies:** types.jl, environment.jl. **Target:** April 20.

---

### Module 12: Belief Updater — `belief_updater.jl`

**From scratch.** Agent's Hamilton (1989) filter for Level 3 only.

`update_agent_belief(belief, log_return, fill_outcome, config)`:
1. **Predict:** ξ_predict = P' × ξ_prior
2. **Return likelihood:** η_j = N(log_return | drift_j, σⱼ√Δt)
3. **Fill likelihood:** ℓ_j = P(fill_outcome | regime = j) via `fill_outcome_likelihood`
4. **Normalize:** ξ_agent = (ξ_predict ⊙ η ⊙ ℓ) / sum(...)

The fill likelihood captures: "if the true regime were j, the market would price at V_market_j (transition-weighted BS from regime j), and the fill probabilities against my quotes would be different." This is how the agent infers where the market is pricing from its own fill patterns.

No market filter needed — the market has perfect knowledge.

**Dependencies:** types.jl, fills.jl, black_scholes.jl. **Target:** April 22.

---

### Module 13: QMDP Solver — `qmdp.jl`

**From scratch.** Solve MDP per regime, weight Q-values by belief: a* = argmax_a Σ_j belief_j · Q_j(s,a).

**Dependencies:** types.jl, value_iteration.jl, belief_updater.jl. **Target:** April 23.

---

## Build Order and Deadline Mapping

### Phase 1: Core Environment — April 9 to 12
Day 1 (Apr 9): types.jl, black_scholes.jl, spot_dynamics.jl.
Day 2 (Apr 10): fills.jl + unit tests.
Day 3 (Apr 11): portfolio.jl + integration tests.
Day 4 (Apr 12): environment.jl + end-to-end smoke test.

### Phase 2: Benchmarks + Value Iteration — April 13 to 14
Day 5 (Apr 13): benchmarks.jl + value_iteration.jl.
Day 6 (Apr 14): evaluation.jl + comparison plots.

### Phase 3: Derivatives Paper — April 15 to 17
Days 7-9: Write and submit derivatives paper.

### Phase 4: Advanced Solvers — April 18 to 24
Days 10-16: MCTS, DQN, belief updater, QMDP, POMCPOW, full L1-L2-L3 evaluation.

### Phase 5: DMU Paper — April 25 to 30
Days 17-22: Write and submit DMU paper.

---

## Risk Mitigation

**If value iteration is too slow:** Reduce discretization first, then approximate methods.
**If DQN doesn't converge:** Use VI policy as behavioral cloning target, or report MCTS only for Level 2.
**If POMDP level is too complex:** Report QMDP only (skip POMCPOW).
**If sequential options cause boundary issues:** Simplify to single long-dated option.
