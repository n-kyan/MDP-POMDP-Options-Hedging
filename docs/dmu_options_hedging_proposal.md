## DMU Final Project Proposal: Optimal Options Hedging Under Uncertainty

Kyan Nelson

CSCI 5264 - Decision Making Under Uncertainty

------

### Problem Statement

Options are financial derivatives whose value depends on an underlying (also called spot) asset (e.g., a stock). A trader holding options is exposed to risk from price movements in that asset. "Hedging" means trading the underlying asset to offset this risk. The standard textbook approach (Black-Scholes) assumes you can hedge continuously and costlessly in a market with constant volatility. In reality, hedging happens at discrete intervals, every trade incurs a cost and the volatility of the spot asset is not constant. This creates a sequential decision problem: **at each timestep, how much should you hedge given your current risk exposure, market conditions, and the cost of trading?**

This project formulates that problem as an MDP, and extends it to a POMDP where a key environment parameter (volatility regime) is hidden. The goal is to outperform Black-Sholes in realistic market conditions.

### Simulation Environment

- **Price dynamics:** The underlying asset price evolves via geometric Brownian motion (Level 1) or a regime-switching model (Levels 2–3). At each timestep, a random return is drawn from a normal distribution parameterized by the current volatility, and the price updates multiplicatively. In the regime-switching model, a hidden Markov chain governs transitions between discrete volatility levels (e.g., low = 12%, high = 35%).
- **Risk exposures (Greeks):** An option's sensitivity to price movements of the spot asset ("delta") and the rate of change of that sensitivity ("gamma") are computed analytically via the Black-Scholes formula at each step, given the current simulated price, option strike/expiry, and volatility. These are standard closed-form expressions.
- **Transaction costs:** Each hedge trade incurs a proportional cost (e.g., 0.1% of trade notional), creating the core tradeoff: hedging reduces risk but costs money.

### MDP Formulation

- **State:**
  - *Portfolio delta:* Net directional risk exposure. Updated analytically each step as price moves, and adjusted by the agent's hedge trades.
  - *Portfolio gamma:* Rate of change of delta. Determines how urgently hedging is needed. High gamma means delta is changing fast.
  - *Volatility Regime:* The true current regime
    - Level 1: Constant and known.
    - Level 2: Observable regime label.
    - Level 3: Becomes a hidden state and is replaced by observations (trailing realized vol, recent returns).
  - *Time to expiry:* Countdown from T to 0.
- **Actions:** Discrete hedge adjustments. Hedge 0%, 25%, 50%, 75%, or 100% of current delta exposure.
- **Transitions:** Price steps forward one period under the stochastic model. Greeks update analytically. Delta adjusts by the hedge action taken.
- **Reward:** Period P&L of the combined options + hedge portfolio, minus transaction costs, with a risk penalty term (variance of returns).

### Benchmark

The baseline is **Black-Scholes delta hedging**. At every timestep, compute the theoretical delta and trade to fully offset it, paying transaction costs each time. This represents the naive "hedge everything always" strategy. Both the learned policy and the benchmark run on the same simulated price paths. Evaluation metrics: cumulative P&L, P&L variance, Sharpe ratio, and hedge trade frequency.

------

## Three Levels of Success

### Level 1 — Minimum Working Example

Single option, **constant known volatility** (standard Geometric Brownian Motion). Discretized state and action spaces. Solve via value iteration from scratch. Compare the optimal policy against the Black-Scholes benchmark.

**Expected result:** The optimal policy hedges less frequently than Black-Scholes when transaction costs are material. It tolerates small delta deviations rather than paying to eliminate them every period.

### Level 2 — Main Approach

Extend to regime-switching **stochastic volatility** (2–3 regimes with Markov transitions). The agent knows the current regime (fully observable). Portfolio holds multiple options, so the agent manages aggregate risk.

Solve using:

- **MCTS**
- **DQN** 

**Compare MCTS vs. DQN vs. Black-Scholes**. Analyze how learned policies adapt to volatility regimes and vary with portfolio risk characteristics.

### Level 3 — Stretch Goal

POMDP formulation: The **volatility regime becomes a hidden state**. The agent observes price returns and trailing realized vol but must infer the regime. A particle filter maintains a belief distribution over regimes. An online POMDP solver plans over the belief space.

Compare against the Level 2 MDP policy (regime known) and DQN (learns to handle uncertainty implicitly) to quantify the cost of partial observability and compare planning vs. learning under hidden state dynamics.

------

### Implementation Breakdown

| Component                                    | Implementation                                 |
| -------------------------------------------- | ---------------------------------------------- |
| Price simulation (GBM, regime-switching)     | From scratch                                   |
| Black-Scholes Greeks and benchmark hedger    | From scratch                                   |
| MDP type definitions and formulation         | From scratch                                   |
| Value iteration (Level 1)                    | From scratch                                   |
| DQN training loop (Level 2)                  | From scratch, using Flux.jl for neural network |
| Particle filter for belief updates (Level 3) | From scratch                                   |
| MCTS solver (Level 2)                        | POMDPs.jl                                      |
| POMDP solver — POMCPOW/DESPOT (Level 3)      | POMDPs.jl                                      |
| Evaluation and comparison framework          | From scratch                                   |
