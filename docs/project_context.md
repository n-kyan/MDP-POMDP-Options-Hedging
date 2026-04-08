# Project Context: Options Market Making Under Uncertainty

## One-Line Summary

A market maker quotes bids and asks on calls and puts at a single strike, manages a dynamically evolving inventory by hedging delta in the underlying, and jointly optimizes spread width and hedge amount. Under partial observability (hidden volatility regime), the agent uses both price returns and fill pattern asymmetry to infer the true regime and adapt its quoting and hedging.

---

## Project Purpose & Stakeholders

This is a joint final project for two CU Boulder courses:

- **ASEN/CSCI 5264 — Decision Making Under Uncertainty (DMU):** Taught by Professor Zachary Sunberg (primary author of POMDPs.jl). Paper due April 30, 4–8 pages, focused on math and algorithms.
- **Applied Derivatives:** Taught by Professor Daniel Brown, who shaped the market-making framing of the project. Paper due April 17, 8–10 pages, focused on finance.

**Kyan's goals beyond the classroom:** This project is intended as a portfolio piece for quantitative trading firm recruiting. Every design choice is filtered through "would this impress a recruiter at an options trading firm?" Kyan is a finance major with CS coursework, relatively new to Julia, and using this project to learn both the DMU course content and Julia simultaneously.

---

## The Market Making Setup

### What the Agent Does

At each timestep, the agent makes two coupled decisions:

1. **Spread width** — how wide to quote around its believed fair value (discrete levels, e.g., {tight, medium, wide, very_wide}). This controls order flow: wider spreads mean fewer fills but more edge per trade; tighter spreads mean more fills but less edge.
2. **Hedge amount** — what target hedge ratio to set (discrete levels, e.g., {0%, 25%, 50%, 75%, 100%, 125%}). This controls delta risk. The agent only hedges delta (trades shares of the underlying), but uses gamma and other Greeks to inform *how much* to hedge.

The combined action space is the cross product of these two dimensions (~30 discrete actions).

### What Happens Each Timestep

1. Agent observes state (inventory, Greeks, moneyness, time to expiry, hedge position, vol belief)
2. Agent chooses action (spread_width, hedge_target)
3. Agent adjusts hedge — trades shares of underlying to reach hedge_target, pays proportional transaction cost
4. Fills arrive — based on spread width and distance from true option value, determine if bid gets hit, ask gets lifted, or nothing happens. If fill occurs, inventory changes by ±1 contract, agent captures half-spread as P&L
5. Spot price moves — one step of GBM (fully observable) or regime-switching (partially observable)
6. Greeks update — recompute delta, gamma for all options in inventory at new price and new tau
7. Reward computed — from mark-to-market + spread capture + hedge P&L - transaction costs - risk penalty

### The Options Being Quoted

- **Calls and puts** at a **single fixed strike** (starts ATM, drifts as spot moves)
- **All options expire on the same day** — the simulation runs for a fixed number of timesteps representing the option's life to expiry (e.g., 1000 steps)
- No multi-strike, no multi-expiry, no rolling into new options
- Even with one strike/expiry, the portfolio is dynamic because fills add and remove inventory every timestep

### Fill Probability Model

- **P(fill per timestep)** is a decreasing function of spread width. Starting with a linear placeholder; exact functional form to be confirmed with Professor Brown.
- **Multiple fills per step:** P(k fills) follows a binomial/geometric model — P(2 fills) = P(fill)², so multi-contract fills are rare.
- **Fill asymmetry carries information (critical for POMDP):** When the agent quotes around its *believed* fair value, the fill probability on each side depends on the distance of that side's quote from the *true* value:
  - P(ask lifted) = f(ask_price − true_value) — decreases as ask gets more expensive relative to truth
  - P(bid lifted) = f(true_value − bid_price) — decreases as bid gets cheaper relative to truth
  - If the agent's belief is correct (centered on true value), fills are symmetric
  - If the agent underestimates vol (underprices the option), ask is cheap → more ask fills, fewer bid fills
  - This asymmetry is an **observation** in the POMDP — the agent can learn that lopsided fills signal mispricing, which updates its belief about the hidden vol regime
- **Fills are symmetric in expectation when correctly priced** — no directional bias in order flow

### P&L Decomposition

Each timestep, P&L comes from four sources:

1. **Mark-to-market on inventory:** Σ(BS_price_new − BS_price_old) across all contracts held
2. **Spread capture on fills:** +(spread/2) for each fill (both bid and ask fills are profitable for the market maker)
3. **Hedge P&L:** hedge_shares × (S_new − S_old)
4. **Hedge transaction cost:** −κ × |shares_traded| × S for any hedge adjustment

Net P&L = mark_to_market + spread_capture + hedge_pnl − hedge_txn_cost

### Spot Price Drift

The spot price uses **risk-neutral drift (μ = r, the risk-free rate)** to prevent the agent from learning directional biases. If the spot had positive expected return, the agent would learn to hold long delta as a directional bet rather than learning to be a good market maker.

---

## MDP / POMDP Formulation

### State Space

The agent needs to know:

| Component | Description |
|---|---|
| **Net inventory** | Number of calls and puts held (can be long or short). Determines portfolio Greeks. |
| **Net portfolio delta (Δ_net)** | Directional risk of combined options inventory + hedge. This is what the agent manages. |
| **Portfolio gamma (Γ)** | How fast delta is changing. High gamma → hedging more urgent. |
| **Moneyness (S/K)** | Where spot is relative to strike. Determines Greek shapes. |
| **Time to expiry (τ)** | Countdown to expiry. Gamma concentrates near strike as τ → 0. |
| **Current hedge position** | Shares of underlying held. Combined with delta gives unhedged risk. |
| **Volatility belief** | In POMDP: belief distribution over regimes. In MDP: actual regime. |

### Action Space

Two coupled discrete decisions (cross product):

- **Spread width:** ~4–5 discrete levels from tight to very wide (exact values TBD with Brown)
- **Hedge target:** {0%, 25%, 50%, 75%, 100%, 125%} of current delta exposure

### Observation Space (POMDP only)

- Everything in state except the true volatility regime
- Spot price return: log(S_new / S_old) — larger returns more likely under high vol
- Fill outcomes: which side got hit (bid, ask, or nothing). Asymmetry signals mispricing → updates vol belief

### Pricing Under Belief Uncertainty

When the agent doesn't know the true vol, it computes a **belief-weighted price** (NOT the BS price at the mean vol):

> price = belief[low] × BS_price(σ_low) + belief[high] × BS_price(σ_high)

This is correct because BS pricing is nonlinear in vol (Jensen's inequality). The agent centers its quotes around this belief-weighted price. As belief updates, the quote center shifts.

### Reward Function

> r = net_pnl − λ × net_pnl²

where net_pnl is the total per-step P&L from all four sources above. The quadratic penalty discourages large P&L swings (proxy for risk aversion). The exact risk measure (quadratic, CVaR, terminal variance, etc.) is to be confirmed with Professor Brown.

---

## Simulation Environment Architecture

### Components (Agent-Independent)

1. **Spot Price Dynamics** — GBM (constant vol) or regime-switching (3-state Markov chain, low <=14%, med 14-21%, high >21%). Built to support arbitrary number of regimes.
2. **Option Pricing & Greeks** — Black-Scholes closed-form for calls and puts. Inputs: S, K, τ, σ, r. Outputs: price, delta, gamma (and vega if needed).
3. **Fill Simulation** — Given agent's quoted spread and the true option value, simulates whether bid/ask/neither gets filled. Asymmetry depends on distance of quotes from truth.
4. **Inventory Tracker** — Bookkeeping for portfolio. Tracks calls/puts held (long and short), computes aggregate Greeks, tracks hedge shares.
5. **P&L Accounting** — Computes per-step P&L from mark-to-market, spread capture, hedge P&L, and transaction costs.

### Build Order

1. Black-Scholes pricing and Greeks (pure math, no dependencies)
2. Spot price simulation (GBM first, regime-switching second)
3. Fill simulation model
4. Inventory tracker and P&L accounting
5. Wire into a single `step!` function
6. POMDPs.jl interface (later, for solvers)

---

## Solver Stack (from DMU Course)

The project applies multiple DMU techniques, building from simple to complex:

| Solver | Type | Level | From Scratch? |
|---|---|---|---|
| Value Iteration | Exact MDP | Fully observable, constant vol | Yes |
| MCTS (DPWSolver) | Online MDP | Fully observable, regime-switching | POMDPs.jl |
| DQN | Model-free RL | Fully observable, regime-switching | Yes (Flux.jl for NN) |
| Exact Bayesian Filter | Belief updater | Hidden regime | Yes |
| Particle Filter | Belief updater | Hidden regime | Yes |
| QMDP | POMDP approximation | Hidden regime | Yes |
| POMCPOW / DESPOT | Online POMDP | Hidden regime | POMDPs.jl |

---

## Three Levels of Complexity

*Note: The exact L1/L2/L3 breakdown has not been finalized since the scope changed to market making. The levels will likely follow this structure but need to be formally defined:*

- **Level 1 (Minimum):** Market making with constant known vol. Agent chooses spread width + hedge amount. GBM spot dynamics. Value iteration solver.
- **Level 2 (Main):** Regime-switching vol, fully observable. Agent adapts spread and hedge to regime. MCTS + DQN solvers.
- **Level 3 (Stretch):** Regime hidden. POMDP with belief updating from price returns AND fill asymmetry. QMDP + online POMDP solvers. Quantify cost of partial observability.

---

## Key Modeling Decisions & Principles

- **Risk-neutral drift (μ = r):** Prevents directional bias in agent behavior
- **Belief-weighted pricing, not mean-vol pricing:** BS is nonlinear in vol → must average prices, not vol (Jensen's inequality)
- **Fill asymmetry as POMDP observation:** Lopsided fills signal mispricing → informs belief about hidden vol regime. This couples the observation model to the agent's action (where it centered quotes), which is valid but makes belief updating more complex.
- **Delta-only hedging:** Agent trades only the underlying. But gamma, and potentially vega, are state variables that inform the hedging decision.
- **Symmetric quoting:** Spread is symmetric around believed fair value. Asymmetric quoting (shifting midpoint to manage inventory) is flagged as a future extension.
- **Single strike, same expiry:** Scope control. Multi-strike and staggered expiry are future extensions.
- **Transaction costs on hedging only:** The agent captures spread on option fills (market maker's edge) and pays proportional costs on underlying trades (hedging cost).

---

## Benchmarks

- **Black-Scholes delta hedging:** Naive "fully hedge every period" baseline. Does not reason about costs.
- **Leland's heuristic:** Adjusts BS delta to account for transaction costs using modified volatility.
- **Whalley-Wilmott:** Asymptotic hedging bandwidth — don't trade unless delta exposure exceeds a threshold.
- **Practitioner benchmarks TBD:** Confirm with Brown whether delta-gamma hedging or other heuristics should be included.

### Evaluation Metrics

- Cumulative P&L (mean and full distribution including tails)
- P&L standard deviation and max drawdown
- Sharpe ratio (mean P&L / std P&L)
- Hedge trade frequency and total transaction costs
- Spread capture vs. hedging cost breakdown
- Cost of partial observability: Level 2 (regime known) vs. Level 3 (regime hidden) performance gap

---

## Open Items Awaiting Professor Brown's Input

1. **Spread width range** — what are realistic levels in dollar terms or as a fraction of option value?
2. **Fill probability function** — linear placeholder for now; need his view on realistic shape and calibration
   1. 

3. **Reward function / risk measure** — is quadratic P&L penalty the right risk measure for market making, or something else (CVaR, Sharpe-based, etc.)?
4. **P&L decomposition completeness** — confirm the four-component decomposition is complete
5. **Adverse selection dynamics** — confirm that fill asymmetry based on distance from true value is the right model for informed vs. uninformed flow
6. **Drift = risk-free rate** — confirm this is the correct treatment for the spot process in a market-making simulation
7. **Calls and puts at one strike, same expiry** — confirm this scoping is sufficient

---

## Tools & Environment

- **Language:** Julia
- **Key packages:** POMDPs.jl ecosystem (QuickPOMDPs, QMDP, POMCPOW, DPWSolver), Flux.jl (DQN), Distributions.jl, Plots.jl
- **Editor:** Typora for Markdown/LaTeX papers
- **Textbook:** Kochenderfer "Algorithms for Decision Making"

---

## Timeline

- **April 17:** Derivatives paper due (finance-focused, 8–10 pages)
- **April 30:** DMU paper due (algorithms-focused, 4–8 pages)
- **Priority:** Core environment and baseline results by ~April 14 to support derivatives paper; solver expansion and POMDP layer for DMU paper by April 30

---

## Writing & Communication Preferences

- Kyan prefers comprehensive explanations over assumptions about prior knowledge
- Homework answers in Markdown with LaTeX; concise, direct first-person writing
- Recruiter legibility is a primary filter for project decisions
- Wants to understand reasoning behind structural decisions, not just accept recommendations
- Multiple layers of abstraction in technical documentation for navigating at different detail levels







hard limit on gamma and Vega

look at constrained POMDP
