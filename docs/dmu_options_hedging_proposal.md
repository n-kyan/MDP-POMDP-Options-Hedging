## DMU Final Project Proposal: Optimal Options Hedging Under Uncertainty

Kyan Nelson

CSCI 5264 — Decision Making Under Uncertainty

------

### Problem Statement

Options are financial derivatives whose value depends on an underlying (also called "spot") asset such as a stock or futures contract. A trader holding options is exposed to risk from price movements in that asset. "Hedging" means trading the underlying asset to offset this risk. The standard textbook approach (Black-Scholes delta hedging) assumes you can hedge continuously and costlessly in a market with constant volatility. In practice, none of these assumptions hold: hedging happens at discrete intervals, every trade incurs a transaction cost, and volatility is stochastic and unobservable. For a derivatives trader, this gap between theory and reality is expensive — suboptimal hedging in the presence of transaction costs can erode a significant portion of an option position's theoretical edge.

This creates a sequential decision problem under uncertainty: **at each timestep, how much should you hedge given your current risk exposure, market conditions, and the cost of trading?** This project formulates that problem as a Markov Decision Process (MDP), solves it with value iteration and online tree search, extends it with deep reinforcement learning, and further generalizes it to a Partially Observable MDP (POMDP) where the volatility regime driving the market is a hidden state that must be inferred from price observations.

------

### Simulation Environment

- **Price dynamics:** The underlying asset price evolves via geometric Brownian motion (Level 1) or a regime-switching model (Levels 2–3). At each timestep, a random return is drawn from a normal distribution parameterized by the current drift and volatility, and the price updates multiplicatively. In the regime-switching model, a hidden Markov chain governs transitions between discrete volatility levels (e.g., low ≈ 12%, high ≈ 35%).

- **Option valuation and Greeks:** The option is valued analytically via Black-Scholes at each step given the current simulated price, strike, time to expiry, and volatility. The option's key risk sensitivities are computed from the standard Black-Scholes closed-form expressions:
  - **Delta (Δ):** Sensitivity of option value to a change in the underlying price. This is the primary risk the agent is hedging.
  - **Gamma (Γ):** Rate of change of delta with respect to the underlying price. High gamma means delta is changing rapidly and hedging is more urgent.

- **Transaction costs:** Each hedge trade incurs a proportional cost (e.g., κ = 0.1% of trade notional), creating the core tradeoff: hedging reduces risk but costs money.

------

### MDP Formulation

**State** — a tuple *s = (δ_net, Γ, m, τ, [σ_regime])*:

| Component | Description |
|---|---|
| Net portfolio delta (δ_net) | Directional risk exposure of the combined option + hedge position. Updated analytically each step as price moves, and adjusted by the agent's hedge trades. |
| Portfolio gamma (Γ) | Rate of change of delta. Determines how urgently hedging is needed — high gamma means delta will shift significantly with the next price move. |
| Moneyness (m = S/K) | Ratio of spot price to strike price. Captures where the option sits relative to its strike, which determines the shape of the Greeks. Also controls the dollar cost of hedging since transaction costs are proportional to notional. Using moneyness rather than raw price makes the policy generalizable across price levels. |
| Time to expiry (τ) | Countdown from T to 0. As expiry approaches, gamma concentrates near the strike and hedging dynamics change significantly. |
| Volatility regime (σ_regime) | **Level 1:** Constant and known (not in state). **Level 2:** Observable regime label (e.g., low/high). **Level 3:** Hidden — removed from the state and replaced by a belief distribution inferred from observations. |

**Actions** — discrete hedge adjustments representing what fraction of the current delta exposure to eliminate: {0%, 25%, 50%, 75%, 100%}. An action of 0% means "do nothing and let exposure ride." An action of 100% means "trade enough underlying shares to zero out net delta." Higher levels may extend the action space to include overshooting (e.g., 125%) to allow pre-hedging of anticipated delta moves near expiry.

**Transitions** — The price steps forward one period under the stochastic model (GBM or regime-switching). The option's Greeks are recomputed analytically at the new price. Net delta adjusts by the hedge action taken. In Level 2+, the volatility regime transitions according to a Markov chain.

**Reward function** — at each timestep *t*, the single-step reward is:

> **r_t = ΔV_option,t + ΔV_hedge,t − κ · |trade_t| · S_t − λ · (ΔPnL_t)²**

where:

- **ΔV_option,t** is the change in the option's Black-Scholes theoretical value from *t* to *t+1* (computed analytically given the true simulation volatility).
- **ΔV_hedge,t** is the mark-to-market P&L on the hedge position (shares of underlying held × price change).
- **κ · |trade_t| · S_t** is the proportional transaction cost of the hedge adjustment. κ is the cost parameter, |trade_t| is the number of shares traded, and S_t is the current spot price.
- **λ · (ΔPnL_t)²** is a risk penalty. ΔPnL_t = ΔV_option,t + ΔV_hedge,t is the net hedging P&L for the period. Squaring it penalizes large P&L swings in either direction, not just losses. The parameter λ controls risk aversion — higher λ produces a more conservative agent that prioritizes hedging consistency over cost minimization. This per-step quadratic penalty is a tractable proxy for the multi-step variance penalty that is standard in hedging cost analysis.

The agent's objective is to maximize the cumulative discounted reward over the life of the option: **max E[Σ_t γ^t r_t]**.

------

### Benchmark

The baseline is **Black-Scholes delta hedging**: at every timestep, compute the theoretical delta and trade to fully offset it, paying transaction costs each time. This represents the naive "hedge everything, every period" strategy that does not reason about transaction costs. Both the learned policies and the benchmark are evaluated on the same simulated price paths. 

**Evaluation metrics:**

- Cumulative P&L (mean and full distribution, including tails)
- P&L standard deviation and max drawdown
- Sharpe ratio (mean P&L / std P&L)
- Hedge trade frequency and total transaction costs paid

Showing distributions (not just means) is critical — risk management is about tails.

------

## Three Levels of Success

### Level 1 — Minimum Working Example

**Setup:** Single European option (call or put), constant known volatility (standard GBM), proportional transaction costs. Discretized state space (δ_net, Γ, moneyness, τ) and discrete action space {0%, 25%, 50%, 75%, 100%}.

**Method:** Solve via **value iteration** implemented from scratch. Sweep over transaction cost levels (κ = 0, 0.05%, 0.1%, 0.2%) to generate a cost-sensitivity analysis.

**Deliverables:**
- Converged value function and policy.
- Comparison of optimal policy vs. Black-Scholes delta hedging across transaction cost levels.
- P&L distribution plots for each method and cost level.
- Policy visualization showing how hedge aggressiveness varies with state (e.g., the agent tolerates larger delta deviations when costs are high, when gamma is low, or when time to expiry is long).

**Expected result:** The optimal policy hedges less frequently than Black-Scholes when transaction costs are material. It tolerates small delta deviations rather than paying to eliminate them every period. At κ = 0, the two policies converge.

------

### Level 2 — Main Approach

**Setup:** Same single-option environment, but now with **regime-switching stochastic volatility** — a 2-state Markov chain governing transitions between low (≈12%) and high (≈35%) volatility. The current regime is **fully observable** (the agent knows which regime it is in). The regime label is added to the state tuple.

**Methods (two approaches, compared against each other and the baseline):**

1. **Monte Carlo Tree Search (MCTS)** via POMDPs.jl — an online planning method that builds a search tree from the current state, using simulated rollouts to estimate action values. Does not require discretizing the full state space up front.

2. **Deep Q-Network (DQN)** implemented from scratch using Flux.jl — a model-free reinforcement learning approach that trains a neural network to approximate Q(s,a) from simulated experience. Learns a policy implicitly through environment interaction.

**Deliverables:**
- Trained DQN policy and MCTS planner, both operational in the regime-switching environment.
- Three-way comparison: **MCTS vs. DQN vs. Black-Scholes** on the same set of simulated paths.
- Analysis of how each learned policy adapts its behavior across volatility regimes (e.g., does the agent hedge more aggressively in high vol because delta is more volatile, or less aggressively because transaction costs are higher in absolute terms?).
- Analysis of planning vs. learning tradeoffs (MCTS computation per step vs. DQN training cost, sample efficiency, policy quality).

**Expected result:** Both MCTS and DQN outperform Black-Scholes under transaction costs. Their policies adapt to the volatility regime — hedging behavior differs between low-vol and high-vol states. MCTS may produce better policies given sufficient computation time per step, while DQN executes faster at inference time after training.

------

### Level 3 — Stretch Goal

**Setup:** Same regime-switching environment as Level 2, but now the **volatility regime is hidden**. The agent cannot see the regime label directly. Instead, it observes only the price process (returns) and must infer the current regime from these observations.

**Methods:**

1. **Exact Bayesian belief updating** — because the hidden state is discrete (2–3 regimes), the belief over regimes is a probability vector that can be updated exactly via Bayes' rule at each timestep:

   > b'(σ') ∝ P(observed return | σ') × Σ_σ T(σ' | σ) × b(σ)

   where T is the regime transition matrix and the observation likelihood is the normal density of the observed return under each volatility level. This is the correct inference tool for a discrete hidden state.

2. **Particle filter** — implemented from scratch as a more general alternative. Approximates the belief distribution with a set of weighted samples. Applied to the same discrete-regime problem, it should converge to the exact Bayesian solution, validating the implementation. This generalizes naturally to continuous hidden states (e.g., if the project were extended to a Heston-style continuous stochastic volatility model).

3. **QMDP** — uses the MDP value function from Level 1/2 and averages it over the current belief state to select actions. This bridges the MDP and POMDP: Q_QMDP(b, a) = Σ_σ b(σ) × Q_MDP(s, a, σ). It is a known approximation that assumes the hidden state will be revealed at the next step, so it tends to undervalue information-gathering actions.

4. **Online POMDP solver (POMCPOW or DESPOT)** from POMDPs.jl — a more sophisticated solver that plans over the full belief space and accounts for the value of future observations.

**Deliverables:**
- Exact belief updater and particle filter, both tracking regime beliefs from observed returns.
- Comparison of belief update methods (exact vs. particle filter convergence).
- QMDP and online POMDP solver, both making hedging decisions under regime uncertainty.
- Quantification of the **cost of partial observability**: compare Level 3 POMDP policies (regime hidden) against Level 2 MDP policies (regime known) and the Black-Scholes baseline. The gap between Level 2 and Level 3 performance measures the value of knowing the true volatility regime.
- Analysis of how the agent hedges when its regime belief is **ambiguous** (e.g., b ≈ [0.5, 0.5]) vs. **confident** (e.g., b ≈ [0.95, 0.05]).

**Expected result:** The POMDP agent hedges more conservatively when uncertain about the regime. The online POMDP solver outperforms QMDP because it accounts for the value of future information. The performance gap between Level 2 (regime known) and Level 3 (regime hidden) quantifies the cost of volatility uncertainty — a concept directly relevant to real trading operations.

------

### Future Extensions (Not in Scope for This Project)

- **Multi-option portfolio hedging:** Extend from a single option to a portfolio of options with different strikes and expiries. The agent would manage aggregate portfolio Greeks rather than single-option exposure.
- **Real market data:** Calibrate the simulation to historical data and test policies on real options. Micro E-mini S&P 500 options offer a liquid, accessible venue with well-defined payoff structures and manageable contract sizes.
- **Continuous stochastic volatility:** Replace the discrete regime-switching model with a continuous volatility process (e.g., Heston model), requiring the particle filter for inference.

------

### Implementation Breakdown

| Component | Implementation |
|---|---|
| Price simulation (GBM, regime-switching) | From scratch |
| Black-Scholes option valuation and Greeks | From scratch |
| Black-Scholes delta hedging benchmark | From scratch |
| MDP state/action/transition formulation | From scratch |
| Value iteration solver (Level 1) | From scratch |
| Reward function and evaluation framework | From scratch |
| DQN training loop (Level 2) | From scratch, using Flux.jl for neural network layers |
| Exact Bayesian belief updater (Level 3) | From scratch |
| Particle filter for belief updates (Level 3) | From scratch |
| QMDP solver (Level 3) | From scratch (uses Level 1/2 MDP value function) |
| MCTS solver (Level 2) | POMDPs.jl |
| Online POMDP solver — POMCPOW/DESPOT (Level 3) | POMDPs.jl |
| Probability distributions | Distributions.jl |
| Visualization and plotting | Plots.jl |
