# Implementation Plan: Options Market Making Under Uncertainty

## Overview

This plan organizes the implementation into modules with explicit dependencies, build order, and deadline targets. Each module specifies what is built from scratch versus what uses existing libraries, and which research paper provides the foundation.

The two deadlines drive the priority ordering. April 17 (derivatives paper) requires a working Level 1 environment, value iteration solver, all analytical benchmarks, and baseline results. April 30 (DMU paper) adds Level 2 (MCTS + DQN) and Level 3 (QMDP + POMCPOW), plus the cost-of-partial-observability analysis.

---

## Module Inventory

### Module 1: Types and Configuration — `types.jl`

**From scratch.** Defines all shared data structures: `OptionContract` (strike K, option type), `HedgingState` (spot S, tau, inventory q, hedge_shares, cash, regime/belief), `EnvironmentState` (extends HedgingState with `market_belief::Vector{Float64}` — the market's returns-only Hamilton filter state; internal to the environment and never passed to the agent as an observation), `MarketMakingAction` (spread_level 1-5, hedge_target 1-6), and `SimConfig` (all simulation parameters including S0, K, r, κ, σ_regimes, transition_matrix, spread_levels, hedge_targets, A, k, φ, Δt, max_inventory Q, n_options_per_episode).

The separation between `HedgingState` (the agent's view) and `EnvironmentState` (the full simulation state) enforces the information barrier. The agent never sees `market_belief` directly; it only experiences its consequences through fill outcomes.

**Dependencies:** None. **Target:** April 9.

---

### Module 2: Black-Scholes Pricing and Greeks — `black_scholes.jl`

**From scratch.** Pure math functions: `bs_price`, `bs_delta`, `bs_gamma`, `bs_vega`, `bs_all` (returning a NamedTuple), internal helper `_d1_d2`. Guard clause returns intrinsic value and limit Greeks when τ ≤ 0 to prevent NaN propagation.

Belief-weighted pricing function: `bs_belief_weighted(S, K, τ, σ_regimes, belief, r, type)` computes the sum of belief_j × V_BS(σ_j) and similarly for Greeks.

**Test suite:** Verify against known values — ATM call at S=K=100, σ=0.20, τ=0.25, r=0.05 should give price ≈ $3.99, delta ≈ 0.53.

**Dependencies:** types.jl. **Target:** April 9.

---

### Module 3: Spot Price Simulation — `spot_dynamics.jl`

**From scratch.** Functions: `step_spot` (single GBM step with risk-neutral drift), `step_regime` (Markov chain transition using Hardy's daily transition matrix), `step_environment` (combined spot + regime step).

Discretization: S_{t+1} = S_t × exp((r - σ²/2)Δt + σ√Δt × Z) where Z ~ N(0,1) and σ = σ_{current_regime}.

**Dependencies:** types.jl. **Target:** April 9.

---

### Module 4: Fill Model — `fills.jl`

**From scratch.** Implements AS (2008) eq. 2.11. Functions: `fill_probability(delta_from_market_value, A, k, Δt)` returning min(1, A·exp(-k·δ)·Δt), `simulate_fills(ask_price, bid_price, V_market, A, k, Δt, rng)` returning (bid_filled, ask_filled), and `compute_quotes(V_believed, spread_level, config)` returning (bid_price, ask_price).

Fill probabilities use `V_market = bs_belief_weighted(S, K, τ, σ_regimes, market_belief, r, type)` — the market's belief-weighted BS price computed from the environment's returns-only Hamilton filter. This is distinct from `V_believed`, which uses the agent's belief. In Levels 1–2 (fully observable), both the agent and the market have the same information, so `V_believed ≈ V_market` and the distinction is moot. In Level 3, the agent's fill-augmented belief may diverge from the market's, producing the fill asymmetry that drives the POMDP observation signal.

`simulate_fills` takes `V_market` as an argument (not the current regime volatility). The caller — `environment.jl` — is responsible for computing `V_market` from the current `market_belief` before calling this function.

**Dependencies:** types.jl, black_scholes.jl. **Target:** April 10.

---

### Module 5: Portfolio and P&L Accounting — `portfolio.jl`

**From scratch.** Tracks inventory (held contracts with sign), aggregate Greeks, hedge shares, and cash. Computes per-step P&L from four sources: mark-to-market, spread capture, hedge P&L, and hedge transaction cost.

Key functions: `update_inventory!`, `execute_hedge!`, `compute_pnl`, `compute_reward` (implementing r = pnl - φ·Δ_net²), and `reset_for_new_option!` (carries cash forward when an option expires and a new one starts).

**Dependencies:** types.jl, black_scholes.jl. **Target:** April 11.

---

### Module 6: Environment Step Function — `environment.jl`

**From scratch.** Wires modules 2–5 into a single `step!` function that executes the full timestep sequence:

1. Compute agent's quotes from `V_believed` and chosen spread level
2. Compute `V_market` from the current `market_belief` via `bs_belief_weighted`
3. Execute hedge (shares traded to reach hedge target, charge transaction cost)
4. Simulate fills using `V_market` (not the true regime vol)
5. Update inventory and cash
6. Step spot price and regime
7. Recompute Greeks
8. Update market belief via returns-only Hamilton filter step (`update_market_belief!`)
9. Compute P&L and reward
10. Check option expiry (if expired, settle and start new option; reset τ, carry cash forward)

The market belief update (step 8) uses only the realized log return from step 6. The agent's belief update (if applicable in Level 3) is performed by the POMDP solver's belief updater in `belief_updater.jl`, not inside `environment.jl` — `environment.jl` only maintains `market_belief`. This keeps the information barrier clean: `environment.jl` manages ground truth and market state; the agent's inference is handled separately.

Also implements `reset!` to initialize a new episode, setting both `market_belief` and (for POMDP) the agent's initial belief to the stationary distribution $(\pi_1, \pi_2)$.

**Dependencies:** All modules 1–5. **Target:** April 12.

---

### Module 7: Analytical Benchmarks — `benchmarks.jl`

**From scratch.** Implements all analytical benchmark strategies.

Spread benchmarks: AS optimal spread (eq. 3.18), AS reservation price (eq. 3.17), and GLF-T asymptotic optimal depths (Proposition 3 closed-form approximation). The GLF-T formulas give inventory-dependent bid/ask depths that are bounded and more realistic than raw AS.

Hedge benchmarks: Leland modified volatility (σ̂² = σ²(1 + √(2/π)·κ/(σ√Δt))), Leland delta (BS delta using σ̂), Whalley-Wilmott bandwidth (H = (3κSe^{-rτ}Γ²/(2γ))^{1/3}), and W-W hedge decision (trade to boundary if outside band).

Includes a `run_benchmark` function that simulates any analytical strategy through the environment and collects P&L statistics for comparison.

**Dependencies:** types.jl, black_scholes.jl, environment.jl. **Target:** April 13.

---

### Module 8: Evaluation and Visualization — `evaluation.jl`

**From scratch.** Functions: `evaluate_policy` (Monte Carlo evaluation), `compare_policies` (runs all policies on same random seeds), `plot_pnl_distributions` (histogram comparison like AS Figure 2), `plot_policy_surface` (learned spread/hedge as function of state), `plot_spread_vs_benchmark` (overlay learned vs. GLF-T), `compute_metrics` (mean, std, Sharpe, max drawdown, fill rates).

**Dependencies:** types.jl, environment.jl, benchmarks.jl. Uses Plots.jl. **Target:** April 14.

---

### Module 9: Value Iteration Solver — `value_iteration.jl`

**From scratch.** Tabular value iteration for the Level 1 discretized MDP. State discretization over ~2,000 states across (Δ_net, moneyness, τ, hedge_position). 30 discrete actions. Functions: `discretize_state`, `value_iteration` (standard Bellman backup with multiple samples per (s,a) for stochastic transitions), `extract_policy`.

**Dependencies:** types.jl, environment.jl. **Target:** April 13.

---

### Module 10: POMDPs.jl Interface — `pomdp_interface.jl`

**From scratch** (interface code) + **POMDPs.jl** (solver infrastructure). Wraps the environment into POMDPs.jl abstract types implementing `gen`, `actions`, `discount`, etc. Enables DPWSolver (MCTS) and POMCPOW.

**Dependencies:** types.jl, environment.jl. Uses POMDPs.jl, QuickPOMDPs.jl. **Target:** April 18.

---

### Module 11: DQN Solver — `dqn.jl`

**From scratch** using Flux.jl. Components: 3-layer MLP (state_dim → 128 → 128 → 30), replay buffer, ε-greedy exploration with linear decay, target network with periodic hard copy, training loop with minibatch TD updates.

**Dependencies:** types.jl, environment.jl. Uses Flux.jl. **Target:** April 20.

---

### Module 12: Belief Updater — `belief_updater.jl`

**From scratch.** Implements both Hamilton (1989) filter variants.

`update_agent_belief!(belief, log_return, fill_outcome, quotes, market_belief, config)`: Full four-step update using returns + fill likelihood. The fill likelihood for regime j is computed as P(fill_outcome | ρ = j, agent's quotes, V_market), where V_market is passed in so the fill likelihood correctly reflects how far the agent's quotes sat relative to the market price under each hypothetical regime. Returns updated belief vector.

`update_market_belief!(market_belief, log_return, config)`: Returns-only three-step update (predict → return likelihood → normalize). No fill likelihood step. This is called by `environment.jl` at every timestep. It represents the information state of a rational market participant who observes only public price data.

Both functions are pure (no mutation of environment state) and return new belief vectors. The caller is responsible for storing the result.

**Fill likelihood computation:** For a given fill outcome (e.g., ask filled, bid not filled), the probability under regime j is:

$$P(\text{ask only} \mid \rho = j) = P_{\text{ask}}(\rho = j) \cdot (1 - P_{\text{bid}}(\rho = j))$$

where each fill probability is computed from `fill_probability` in `fills.jl` using $V_{\text{market}} = \sum_i \xi_i^{\text{market}} V_{\text{BS}}(\sigma_i)$. Note that the fill likelihood depends on the *current* market belief (which determines V_market), creating a coupling between the two filters. This coupling is resolved by always computing V_market from the *prior* market belief (before the current step's return is incorporated) to avoid lookahead bias.

**Dependencies:** types.jl, fills.jl, black_scholes.jl. **Target:** April 22.

---

### Module 13: QMDP Solver — `qmdp.jl`

**From scratch.** Solves the underlying MDP for each regime via value iteration, then selects actions by weighting Q-values with belief: a* = argmax_a Σ_j belief_j · Q_j(s,a).

**Dependencies:** types.jl, value_iteration.jl, belief_updater.jl. **Target:** April 23.

---

## Build Order and Deadline Mapping

### Phase 1: Core Environment — April 9 to 12

Day 1 (Apr 9): types.jl, black_scholes.jl, spot_dynamics.jl. Day 2 (Apr 10): fills.jl + unit tests for all Day 1 modules. Day 3 (Apr 11): portfolio.jl + integration tests. Day 4 (Apr 12): environment.jl + end-to-end smoke test with random policy.

**Milestone:** Can run step!(state, random_action, config, rng) in a loop and collect P&L.

### Phase 2: Benchmarks + Value Iteration — April 13 to 14

Day 5 (Apr 13): benchmarks.jl + value_iteration.jl. Day 6 (Apr 14): evaluation.jl + generate comparison plots.

**Milestone:** Comparison table showing RL agent vs. AS, GLF-T, Leland, Whalley-Wilmott across 1,000 episodes.

### Phase 3: Derivatives Paper — April 15 to 17

Day 7 (Apr 15): Write model description and literature review sections. Day 8 (Apr 16): Write results section from Phase 2 outputs and discussion. Day 9 (Apr 17): Final editing, submit.

### Phase 4: Advanced Solvers — April 18 to 24

Day 10 (Apr 18): pomdp_interface.jl + test with DPWSolver (MCTS). Day 11 (Apr 19): Run MCTS on Level 2, collect results. Day 12 (Apr 20): dqn.jl training loop + initial training runs. Day 13 (Apr 21): DQN hyperparameter tuning, Level 2 DQN results. Day 14 (Apr 22): belief_updater.jl (Hamilton filter). Day 15 (Apr 23): qmdp.jl + test POMCPOW on Level 3. Day 16 (Apr 24): Full evaluation — L1 vs L2 vs L3, cost-of-partial-observability analysis.

**Milestone:** Complete results for all three levels.

### Phase 5: DMU Paper — April 25 to 30

Days 17–19 (Apr 25–27): Write paper focusing on MDP formulation, solver comparison, POMDP analysis. Days 20–21 (Apr 28–29): Figures and editing. Day 22 (Apr 30): Submit.

---

## Risk Mitigation

**If value iteration is too slow:** Reduce discretization granularity first, then switch to approximate methods.

**If DQN doesn't converge:** Start with value iteration policy as behavioral cloning target, then fine-tune. Alternatively, report only MCTS results for Level 2.

**If POMDP level is too complex:** Report QMDP only (skip POMCPOW). QMDP is fast because it reuses MDP value functions.

**If sequential options cause boundary issues:** Simplify to a single long-dated option (1 year) for initial results.
