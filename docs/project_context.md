# Project Context: Options Market Making Under Uncertainty

## One-Line Summary

A market maker quotes bids and asks on calls and puts at a single strike, manages inventory by delta-hedging the underlying, and jointly optimizes spread width and hedge ratio using MDP/RL/POMDP solvers. The true volatility regime is hidden, making this a POMDP where the agent infers the regime from price returns and fill asymmetry.

---

## Project Purpose and Stakeholders

This is a joint final project for two CU Boulder courses:

- **ASEN/CSCI 5264 — Decision Making Under Uncertainty (DMU):** Taught by Professor Zachary Sunberg (primary author of POMDPs.jl). Paper due **April 30**, 4–8 pages, focused on algorithms and DMU course content.
- **Applied Derivatives:** Taught by Professor Daniel Brown, who shaped the market-making framing. Paper due **April 17**, 8–10 pages, focused on the financial modeling and derivatives theory.

**Recruiting goal:** This project is intended as a portfolio piece for quantitative trading firm applications. Every design choice is filtered through "would this impress a recruiter at an options trading firm?" The project is implemented in Julia.

---

## Research Foundation

Every modeling component maps to a published paper. No parameters are invented.

| Component | Paper | What We Take |
|---|---|---|
| Fill intensity model | Avellaneda & Stoikov (2008) | $\lambda(\delta) = Ae^{-k\delta}$, simulation framework |
| Spread benchmark (foundational) | Avellaneda & Stoikov (2008) eq. 3.18 | Optimal spread formula under constant vol |
| Spread benchmark (primary) | Guéant, Lehalle & Fernandez-Tapia (2013) Prop. 3 | Asymptotic bid/ask depths with bounded inventory |
| Regime-switching parameters | Hardy (2001) Table 1 | σ₁ = 12.1%, σ₂ = 26.9%, transition matrix |
| Regime-switching methodology | Hamilton (1989) | Markov-switching model foundation |
| Hedge benchmark (simple) | Leland (1985) | Modified volatility $\hat{\sigma}$ |
| Hedge benchmark (optimal) | Whalley & Wilmott (1997) | No-trade bandwidth $H$ |
| Hedge theory (justification) | Davis, Panas & Zariphopoulou (1993) | Proof that no-trade band is optimal |
| Reward function structure | Cartea, Jaimungal & Penalva (2015) Ch. 10 | Running penalty formulation, adapted from $q^2$ to $\Delta_{\text{net}}^2$ |
| POMDP belief update | Hamilton (1989) filter | 4-step recursive Bayesian update |
| Fill asymmetry as observation | Tsaknaki, Lillo & Mazzarisi (2024) | Order flow carries regime information |
| Market belief model | Hamilton (1989) filter | Market runs returns-only filter; agent's fill signal is private information advantage |
| Novelty confirmation | Fang & Xu (2023), Shi et al. (2024) | No paper jointly learns spread + hedge for options |
| RL hedging landscape | Pickard & Lawryshyn (2023) survey | Confirms no RL paper combines spread + hedge |

**Novelty claim:** No existing paper jointly *learns* both spread width and hedge ratio for options under regime-switching volatility. Fang & Xu (2023) learn spreads but hardcode the hedge at BS delta. Shi, Tang & Zhou (2024) jointly learn spread and hedge but for FX spot, not options. No RL market-making paper uses regime-switching dynamics. The project occupies a genuine gap at the intersection of four independently explored directions: (1) joint spread + hedge learning, (2) options as asset class, (3) regime-switching volatility, (4) POMDP regime inference via fill asymmetry. The dual-filter architecture — where the market prices using public return information while the agent augments its belief with private fill outcomes — adds a fifth dimension of novelty: a principled model of *why* the agent's POMDP advantage exists.

---

## Environment Design

### Spot Price Dynamics

The underlying follows a 2-state regime-switching geometric Brownian motion with **risk-neutral drift** ($\mu = r$ in both regimes) to prevent the agent from learning directional biases. Regimes differ only in volatility.

**Parameters from Hardy (2001), converted to daily frequency:**

| Parameter | Regime 1 (Low Vol) | Regime 2 (High Vol) |
|---|---|---|
| σ (annualized) | **12.1%** | **26.9%** |
| σ (daily) | 0.76% | 1.70% |
| μ (daily) | r/252 | r/252 |

**Daily transition matrix** (converted from Hardy's monthly estimates via $p_{\text{daily}} \approx 1 - (1 - p_{\text{monthly}})^{1/21}$):

|  | → Regime 1 | → Regime 2 |
|--|---|---|
| From Regime 1 | 0.9982 | **0.0018** |
| From Regime 2 | **0.0022** | 0.9978 |

**Stationary distribution:** π₁ ≈ 0.55 (low vol), π₂ ≈ 0.45 (high vol). Expected regime durations are ~556 trading days (low vol) and ~455 trading days (high vol).

**Discretization:** At each timestep, $S_{t+1} = S_t \exp\left((r - \sigma^2/2)\Delta t + \sigma\sqrt{\Delta t} \cdot Z\right)$ where $Z \sim N(0,1)$ and $\sigma = \sigma_{\rho_t}$ is the current regime's volatility.

### Sequential Option Lifetimes

Because regime switches occur on the order of 1–2 years while a single option lives for ~3 months (~63 trading days), a single option lifetime will rarely see a regime switch. To ensure the agent encounters regime switches during training, the simulation runs **multiple sequential option lifetimes**. When the current option expires, the agent's cash P&L carries forward, a new ATM option is initiated at the current spot price (new K = round(S)), and market making continues. This approach is more realistic than artificially accelerating regime switches — real market makers continuously roll their books.

A single training episode consists of **4–8 sequential options** (~1–2 years of simulated time), ensuring 1–3 regime switches are likely per episode.

### Option Pricing and Greeks

Black-Scholes closed-form for European calls and puts. Inputs: $S$, $K$, $\tau$, $\sigma$, $r$. Outputs: price, delta, gamma, vega, theta.

**Pricing under belief uncertainty (POMDP):** The agent computes a belief-weighted price:

$$V_{\text{believed}} = \sum_j \xi_j^{\text{agent}} \times V_{\text{BS}}(\sigma_j)$$

This is correct because BS pricing is nonlinear in σ (Jensen's inequality prevents using mean vol). Greeks are similarly belief-weighted for the agent's hedging decisions.

**Market consensus price ($V_{\text{market}}$):** The fill model — which represents the rest of the market — does not use the true current-regime vol. Using the true vol would mean the market is modeled as clairvoyant while the agent must infer the regime, which inverts the real-world relationship where market consensus is the *more* informed price. Instead, the market is modeled as running its own Hamilton filter updated only on publicly observable return information:

$$V_{\text{market}} = \sum_j \xi_j^{\text{market}} \times V_{\text{BS}}(\sigma_j)$$

The market belief $\xi^{\text{market}}$ is updated each step using only the return likelihood (step 2 of the Hamilton filter). It does not receive the fill asymmetry signal because the rest of the market does not observe the agent's order flow. The agent's belief $\xi^{\text{agent}}$ is updated using both returns *and* fill outcomes — giving the agent a genuine private information advantage. This dual-filter architecture is the structural justification for why the POMDP formulation produces value over the fully observable benchmark: the agent's fill signal is informative precisely because the market cannot see it.

### Fill Model

From Avellaneda & Stoikov (2008) eq. 2.11. Fill intensity decays exponentially with distance from the market consensus value:

$$\lambda(\delta) = A \cdot e^{-k\delta}$$

Over a discrete timestep $\Delta t$, fills on each side are independent Bernoulli trials:

$$P(\text{ask fill}) = \min\left(1,\; A \cdot e^{-k(\text{ask\_price} - V_{\text{market}})} \cdot \Delta t\right)$$
$$P(\text{bid fill}) = \min\left(1,\; A \cdot e^{-k(V_{\text{market}} - \text{bid\_price})} \cdot \Delta t\right)$$

This is equivalent to the Poisson model when $\lambda \cdot \Delta t \ll 1$. Four possible outcomes per step: {no fill, bid only, ask only, both}. Each fill is ±1 contract.

**Adverse selection mechanism:** The agent quotes around $V_{\text{believed}}$ (its own belief-weighted price). Fill probabilities are computed using $V_{\text{market}}$ (the market's returns-only belief-weighted price). When the agent's belief diverges from the market's — for example, because the agent's fill signal has updated it toward the true regime faster than returns alone can inform the market — the agent's quotes will be mis-centered relative to $V_{\text{market}}$, producing asymmetric fills. This is the primary POMDP observation signal.

Crucially, fill asymmetry is a *private* signal: the agent observes which side fills more, and can use this to accelerate its belief update beyond what public return information alone provides. The market, updating only on returns, cannot exploit this channel. This is the economic content of the POMDP layer.

**Parameter calibration:** $A$ and $k$ must be calibrated so that the 5 discrete spread levels produce fill probabilities ranging from ~80% (tightest) to ~10% (widest) per side per step. AS uses $A = 140$, $k = 1.5$ for stock market making; for options (priced at ~\$4 vs. \$100 stock), $k$ will need to be larger (around $k = 5$–$8$) to produce meaningful differentiation across the \$0.05–\$0.80 half-spread range. Exact $k$ determined by simulation tuning.

### Spread Width Action Levels

Five fixed dollar half-spread levels, calibrated to the ATM option's initial characteristics:

| Level | Half-spread δ | Calibration Basis | Fill Rate (approx) |
|---|---|---|---|
| 1 | \$0.05 | ≈ 1× hedge cost ($\Delta \cdot S \cdot \kappa$) | ~80% |
| 2 | \$0.10 | ≈ 0.5× initial vega | ~55% |
| 3 | \$0.20 | ≈ 1× initial vega | ~35% |
| 4 | \$0.40 | ≈ 2× initial vega | ~18% |
| 5 | \$0.80 | ≈ 4× initial vega | ~8% |

These remain fixed throughout each option lifetime. The dollar amounts ensure consistent learning dynamics — the agent always knows what each action "costs" in terms of fill probability and edge per fill.

### Hedge Action Levels

Six discrete hedge ratio targets as a fraction of current net delta exposure:

{0%, 25%, 50%, 75%, 100%, 125%}

The agent specifies a target; the environment trades shares to reach that target and charges proportional transaction cost $\kappa \cdot |\text{shares\_traded}| \cdot S$ on the shares traded.

### Combined Action Space

5 spread levels × 6 hedge levels = **30 discrete actions**.

---

## MDP / POMDP Formulation

### State Space

| Component | Description | Discretization |
|---|---|---|
| Net inventory $q$ | Calls + puts held (signed) | Integer, bounded ±Q |
| Net portfolio delta $\Delta_{\text{net}}$ | Option deltas + hedge shares | Continuous (discretized for VI) |
| Moneyness $S/K$ | Spot relative to strike | ~10 bins |
| Time to expiry $\tau$ | Countdown to current option expiry | ~10 bins |
| Current hedge position | Shares of underlying held | Continuous (discretized for VI) |
| Volatility regime (MDP) | Known regime index {1, 2} | 2 values |
| Agent belief vector (POMDP) | $P(\rho = j)$ for each regime — updated via returns + fills | Discretized on [0, 1] |
| Market belief vector (env only) | $\xi^{\text{market}}_j$ — updated via returns only, never observed by agent | Continuous, internal to environment |

The market belief vector is an internal environment variable, not part of the agent's observation. It determines $V_{\text{market}}$ for fill probability computation but is never directly accessible to the agent.

### Observation Space (POMDP only)

The agent observes everything in the state except the true regime and the market belief vector, plus two signals:

1. **Log return:** $\log(S_{t+1}/S_t)$. Larger magnitude returns are more likely under high vol.
2. **Fill outcome:** {no fill, bid only, ask only, both}. Asymmetry signals that the agent's quotes are mis-centered relative to $V_{\text{market}}$, which implies the agent's belief diverges from the market's. This is a private signal — the market cannot observe the agent's fill outcomes.

### Reward Function

$$r_t = \text{net\_pnl}_t - \phi \cdot \Delta_{\text{net},t}^2$$

where $\text{net\_pnl}$ is the sum of four components:

1. **Mark-to-market on inventory:** $\sum(\text{BS\_price\_new} - \text{BS\_price\_old})$ across all contracts
2. **Spread capture on fills:** $+\delta$ for each fill (half-spread captured per side)
3. **Hedge P\&L:** $\text{hedge\_shares} \times (S_{\text{new}} - S_{\text{old}})$
4. **Hedge transaction cost:** $-\kappa \cdot |\text{shares\_traded}| \cdot S$

The $\phi \cdot \Delta_{\text{net}}^2$ penalty discourages unhedged directional exposure, adapted from CJP's (2015) running inventory penalty to account for the nonlinear risk profile of options. Under Gaussian returns, this is equivalent to CARA exponential utility with $\gamma = 2\phi$.

### Dual Belief Update (POMDP, Hamilton Filter)

Two parallel Hamilton filters run each timestep. Both use the same four-step recursive structure but receive different information.

**Agent belief update** (uses returns + fills):

1. **Predict:** $\hat{\xi}_{t|t-1}^{\text{agent}} = P^\top \cdot \hat{\xi}_{t-1|t-1}^{\text{agent}}$
2. **Return likelihood:** $\eta_{j,t} = (2\pi\sigma_j^2 \Delta t)^{-1/2} \exp\left(-(r_t - (r - \sigma_j^2/2)\Delta t)^2 / (2\sigma_j^2 \Delta t)\right)$
3. **Fill likelihood:** Multiply by $P(\text{fill\_outcome} \mid \rho = j,\; \text{agent's quotes},\; V_{\text{market}})$ — this is the private observation channel
4. **Normalize:** $\hat{\xi}_{t|t}^{\text{agent}} = (\hat{\xi}_{t|t-1}^{\text{agent}} \odot \eta_t^{\text{agent}}) / (\hat{\xi}_{t|t-1}^{\text{agent}\top} \cdot \eta_t^{\text{agent}})$

**Market belief update** (uses returns only):

1. **Predict:** $\hat{\xi}_{t|t-1}^{\text{market}} = P^\top \cdot \hat{\xi}_{t-1|t-1}^{\text{market}}$
2. **Return likelihood:** same $\eta_{j,t}$ as above
3. *(No fill likelihood step — the market does not observe the agent's fills)*
4. **Normalize:** $\hat{\xi}_{t|t}^{\text{market}} = (\hat{\xi}_{t|t-1}^{\text{market}} \odot \eta_t) / (\hat{\xi}_{t|t-1}^{\text{market}\top} \cdot \eta_t)$

Both filters are O(K²) per step for K regimes. The fill likelihood step in the agent's filter is the only computational difference. Both filters are initialized at the stationary distribution $(\pi_1, \pi_2)$ at the start of each episode.

---

## Solver Stack

| Solver | Type | Level | From Scratch? |
|---|---|---|---|
| Value Iteration | Exact MDP | L1: Constant vol | Yes |
| MCTS (DPWSolver) | Online MDP | L2: Known regime | POMDPs.jl |
| DQN | Model-free RL | L2: Known regime | Yes (Flux.jl for NN) |
| Hamilton Filter | Belief updater | L3: Hidden regime | Yes |
| QMDP | POMDP approx | L3: Hidden regime | Yes |
| POMCPOW | Online POMDP | L3: Hidden regime | POMDPs.jl |

---

## Analytical Benchmarks

### Spread Benchmarks

1. **Symmetric quoting (naive):** Fixed spread centered on mid-price, ignoring inventory. AS's Table 1 baseline.
2. **Avellaneda-Stoikov inventory strategy:** Reservation price $r = s - q\gamma\sigma^2(T-t)$ with optimal spread $\delta^a + \delta^b = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln(1 + \gamma/k)$.
3. **Guéant-Lehalle-Fernandez-Tapia (primary benchmark):** Asymptotic optimal depths with bounded inventory, from GLF-T (2013) Proposition 3.

### Hedge Benchmarks

1. **Black-Scholes delta hedge (naive):** Fully hedge to BS delta every period. No cost awareness.
2. **Leland modified delta:** Replace $\sigma$ with $\hat{\sigma}^2 = \sigma^2(1 + \sqrt{2/\pi} \cdot \kappa / (\sigma\sqrt{\Delta t}))$.
3. **Whalley-Wilmott bandwidth:** No-trade region $H = (3\kappa S e^{-r(T-t)} \Gamma^2 / (2\gamma))^{1/3}$. Hedge only when position exits $[\Delta_{\text{BS}} - H,\; \Delta_{\text{BS}} + H]$.

---

## Three Levels of Complexity

### Level 1 — Minimum Working Example

Fully observable MDP with constant volatility (σ = 20% annualized, the stationary-average of the 2-regime model). The agent knows the true (constant) vol at all times. Value iteration over discretized state space (~2,000 states). Benchmarked against AS and GLF-T optimal strategies and all three hedge benchmarks. **Purpose:** Validate the environment and confirm the RL agent matches analytical solutions.

### Level 2 — Main Approach

Fully observable MDP with regime-switching volatility. The true regime is included in the state. The agent must adapt spread and hedge behavior across both regimes. Solvers: MCTS (DPWSolver from POMDPs.jl) and DQN (from scratch using Flux.jl). **Purpose:** Show that the RL agent outperforms constant-vol benchmarks by adapting to regime changes. AS and GLF-T benchmarks are computed per-regime and compared.

### Level 3 — Stretch Goal

POMDP with hidden volatility regime. The agent maintains a belief distribution over 2 regimes updated each step via the full Hamilton filter (returns + fills). The environment simultaneously maintains a market belief updated via returns only, which determines $V_{\text{market}}$ for fill probability computation. Solvers: QMDP (from scratch), POMCPOW (POMDPs.jl). **Key deliverables:** (1) Quantify the *value of private information* — the performance gap between an agent that uses only returns to update its belief versus one that also uses fill asymmetry. (2) Quantify the *cost of partial observability* — the performance gap between L2 (regime known) and L3 (regime hidden). The dual-filter design makes both analyses clean: the market belief tracks the "returns-only" agent, so comparing agent P&L to a returns-only policy isolates the value of the fill signal directly.

---

## Evaluation Metrics

- Cumulative P&L: mean, standard deviation, full distribution including tails
- Sharpe ratio: mean P&L / std P&L
- Maximum drawdown
- Hedge trade frequency and total transaction costs paid
- Spread capture vs. hedging cost breakdown
- Fill rate and fill asymmetry statistics
- Cost of partial observability: L2 vs. L3 performance gap
- Policy visualization: learned spread and hedge thresholds vs. analytical benchmarks

---

## Simulation Parameters (Defaults)

| Parameter | Value | Source |
|---|---|---|
| Initial spot $S_0$ | 100 | AS (2008) |
| Strike $K$ | 100 (ATM, reset each option) | Design choice |
| Time to expiry per option | 63 trading days (~3 months) | Design choice |
| Options per episode | 4–8 (sequential) | Design choice |
| Risk-free rate $r$ | 0.05 (5% annualized) | Standard |
| Transaction cost $\kappa$ | 0.001 (10 bps proportional) | Standard |
| Risk aversion $\phi$ | Tuned (start ~0.01) | CJP framework |
| Fill intensity $A$ | Tuned (start ~140) | AS (2008) baseline |
| Fill decay $k$ | Tuned (target 5–8 for options) | Calibrated |
| Timestep $\Delta t$ | 1/252 (1 trading day) | Standard |
| Max inventory $Q$ | 10 contracts per side | Design choice |

---

## Timeline

- **April 14:** Core environment complete, Level 1 value iteration running, baseline results
- **April 17:** Derivatives paper submitted (finance focus, 8–10 pages)
- **April 24:** Level 2 (MCTS + DQN) and Level 3 (QMDP + POMCPOW) complete
- **April 30:** DMU paper submitted (algorithms focus, 4–8 pages)
