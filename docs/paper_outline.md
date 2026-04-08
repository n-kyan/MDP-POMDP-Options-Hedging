# Paper Outlines and Abstract

This document contains outlines for both papers and a shared abstract.

---

## Shared Abstract

**Options Market Making Under Uncertainty: A POMDP Approach to Joint Spread and Hedge Optimization**

We study an options market maker who jointly optimizes bid-ask spread width and delta-hedge ratio under stochastic, partially observable volatility. The market maker quotes calls and puts at a single strike, accumulates inventory through stochastic fills, and hedges directional exposure by trading the underlying with proportional transaction costs on each hedge adjustment. We model order arrival using the exponential fill intensity of Avellaneda and Stoikov (2008) and volatility dynamics as a 2-state Markov regime-switching process calibrated to S&P 500 returns following Hardy (2001). Critically, the market consensus option price — against which the agent's fill probabilities are computed — is modeled as a belief-weighted Black-Scholes price derived from a Hamilton (1989) filter updated on publicly observable return data only. At the highest level of complexity, the true volatility regime is also hidden from the agent, creating a partially observable Markov decision process (POMDP) in which the agent maintains a separate belief updated on both returns and its own fill outcomes. Because the market cannot observe the agent's order flow, fill asymmetry (i.e., lopsided execution rates signaling mispricing relative to the market consensus) constitutes a private information signal that accelerates regime inference beyond what public returns alone permit. We compare three levels of environmental complexity: constant known volatility (solved via value iteration), regime-switching with known regime (solved via Monte Carlo tree search and deep Q-networks), and regime-switching with hidden regime (solved via QMDP and online POMDP methods). The learned policies are benchmarked against the Guéant-Lehalle-Fernandez-Tapia (2013) optimal market-making strategy for spread-setting and the Whalley-Wilmott (1997) asymptotic hedging bandwidth for delta management. To our knowledge, this is the first work to jointly learn both spread and hedge policies for options under regime-switching volatility, the first to use fill asymmetry as a POMDP observation for volatility regime inference, and the first to model the market consensus price as a dynamically filtered belief rather than a point estimate of current-regime volatility.

---

## Paper 1: Derivatives Course (Due April 17, 8–10 pages)

**Title:** Options Market Making Under Regime-Switching Volatility: A Simulation-Based Analysis of Joint Spread and Hedge Optimization

**Emphasis:** Financial modeling, derivatives theory, market microstructure, benchmark comparisons.

### Section 1: Introduction (1 page)

Motivate the problem from the perspective of an options market maker. The core tension: wider spreads earn more edge per fill but reduce order flow, while frequent hedging reduces risk but incurs transaction costs. Under stochastic volatility, the optimal trade-off is state-dependent. Existing analytical solutions (AS, GLF-T) assume constant or known volatility — when volatility is uncertain, these solutions are suboptimal and learned policies can improve.

State the contribution: a simulation framework for options market making that extends the Avellaneda-Stoikov stock market-making model to options with regime-switching volatility, benchmarked against analytical optimal strategies. Highlight the dual-filter market model as a key design choice: rather than treating the market as clairvoyant about the current volatility regime, we model the market as a Bayesian agent that updates on public return information, which is the more realistic assumption and the one that gives fill asymmetry its economic content.

### Section 2: Literature Review (1.5 pages)

Organize around three streams. The market-making stream covers Ho & Stoll (1981), Avellaneda & Stoikov (2008), Guéant et al. (2013), and Cartea, Jaimungal & Penalva (2015). The hedging-under-costs stream covers Black-Scholes, Leland (1985), Whalley & Wilmott (1997), and Davis et al. (1993). The regime-switching stream covers Hamilton (1989), Hardy (2001), and Ang & Timmermann (2012). Conclude by noting the gap: no existing work combines market-making spread optimization with learned hedging for options under stochastic volatility, and no prior simulation model of options market making treats the market consensus price as a dynamically filtered belief.

### Section 3: Model (2.5 pages)

**3.1 Environment.** GBM spot dynamics with regime-switching volatility. Hardy (2001) calibrated parameters. Risk-neutral drift. Black-Scholes pricing and Greeks. Sequential option lifetimes.

**3.2 Market-making mechanism.** Fill model from AS: λ(δ) = Ae^{-kδ}. Discrete spread levels calibrated to hedge cost and vega.

The key modeling choice is what constitutes the "true value" against which fill probabilities are computed. Using the current-regime Black-Scholes price would mean the market is perfectly informed about the volatility regime — effectively clairvoyant — while the agent must infer it. This inverts the real-world relationship in which market consensus is the more informed price. Instead, we model the market as running a Hamilton filter updated only on publicly observable log returns, producing a belief-weighted consensus price:

$$V_{\text{market}} = \sum_j \xi_j^{\text{market}} \cdot V_{\text{BS}}(\sigma_j)$$

Fill probabilities are computed against $V_{\text{market}}$, not against the true-regime price. The agent quotes around its own belief-weighted price $V_{\text{believed}}$. When the agent's belief diverges from the market's — for example, because a recent string of large returns has moved the market toward high-vol while the agent has not yet updated — the agent's quotes are mis-centered and fills become asymmetric. The four-component P&L decomposition follows.

**3.3 Hedging mechanism.** Discrete hedge ratio targets. Proportional transaction costs on the underlying. The interaction between spread and hedge decisions.

**3.4 Reward function.** Running delta penalty r = pnl - φ·Δ_net², adapted from CJP's (2015) inventory penalty. Justification via CARA equivalence under Gaussian returns.

### Section 4: Analytical Benchmarks (1 page)

Present the three spread benchmarks (symmetric, AS, GLF-T) and three hedge benchmarks (naive BS delta, Leland, Whalley-Wilmott) with their exact formulas. Explain what each captures and what it misses. Note that all three spread benchmarks assume the market correctly prices at the current-regime vol — they serve as upper bounds on spread performance and a useful calibration target for Level 1, but are strictly misspecified for Levels 2 and 3.

### Section 5: Solution Method (0.5 pages)

Brief description of value iteration over discretized state space. Mention that more advanced solvers (MCTS, DQN, POMDP) are applied in companion work — keep this short since the DMU paper covers algorithms in depth.

### Section 6: Results (2 pages)

**6.1 Environment validation.** Under constant vol, the RL agent's spread policy matches GLF-T and its hedge policy matches the Whalley-Wilmott bandwidth. Because $V_{\text{market}}$ converges to the true price when both the market and the agent have no regime uncertainty, the dual-filter architecture is consistent with the analytical benchmarks in the degenerate case. This validates the simulation.

**6.2 Regime-switching advantage.** Under regime-switching, the RL agent outperforms constant-vol benchmarks by adapting spread and hedge to the current regime. Show P&L distribution comparison (histograms like AS Figure 2), Sharpe ratios, and hedge frequency.

**6.3 Sensitivity analysis.** Transaction cost sensitivity surface showing how optimal policy changes with κ. Spread-hedge coordination: how the agent trades off tighter spreads (more fills, more delta risk) against more aggressive hedging (more cost).

### Section 7: Discussion and Future Work (0.5 pages)

Discuss limitations: stylized simulation, no empirical validation, single strike. The dual-filter market model is itself a simplification — real market consensus reflects heterogeneous participants with different models and risk appetites, not a single Bayesian agent. Future directions: multi-strike, continuous action spaces (DDPG), real data calibration, the full POMDP extension (the subject of the DMU companion paper).

---

## Paper 2: DMU Course (Due April 30, 4–8 pages)

**Title:** Joint Spread and Hedge Optimization for Options Market Making: From MDPs to POMDPs Under Hidden Volatility Regimes

**Emphasis:** MDP/POMDP formulation, solver comparison, cost of partial observability, course content application.

### Section 1: Introduction (0.5 pages)

Frame as: options market making is a natural application of DMU course content. The problem involves sequential decision-making under uncertainty (MDP), exploration-exploitation (RL), and hidden state inference (POMDP). Briefly state the three levels of complexity and the two key quantitative results: the cost of partial observability (the performance gap between knowing and not knowing the regime) and the value of private information (the additional performance gained by incorporating fill asymmetry into the agent's belief update, relative to using returns alone).

### Section 2: Problem Formulation (1.5 pages)

**2.1 MDP formulation.** State space (Δ_net, moneyness, τ, hedge position, regime), action space (5 spread × 6 hedge = 30 actions), transition dynamics (regime-switching GBM + fill model), reward function (pnl - φ·Δ_net²). Formal notation matching Kochenderfer's textbook.

**2.2 POMDP extension.** Hidden state (volatility regime). Observation model: log return (magnitude is a public signal of regime) and fill outcome (a private signal — the market cannot observe the agent's fills).

The distinction between public and private observations motivates the dual-filter architecture. The environment maintains a *market belief* $\xi^{\text{market}}$ updated each step on log returns only — this determines the market consensus price $V_{\text{market}}$ against which fill probabilities are computed. The agent maintains a separate *agent belief* $\xi^{\text{agent}}$ updated on both log returns and fill outcomes. The two filters run in parallel, diverging when the agent's fill signal provides information that the return signal has not yet delivered.

This architecture makes fill asymmetry genuinely informative in a formal sense: because the market prices at $V_{\text{market}}$ (not the true-regime price), the agent's quotes can be correctly centered relative to the true regime while still diverging from $V_{\text{market}}$, producing asymmetric fills. The direction and magnitude of the asymmetry are a function of how far the agent's belief has advanced ahead of the market's, giving the agent a quantifiable private information advantage.

Belief update for both filters: four-step Hamilton filter. Agent filter uses all four steps including the fill likelihood $P(\text{fill\_outcome} \mid \rho = j)$. Market filter skips the fill likelihood step. Both initialize at the stationary distribution $(\pi_1, \pi_2)$.

### Section 3: Solvers (1.5 pages)

**3.1 Value iteration** (Level 1). Discretized state space, Bellman backup, convergence. From scratch.

**3.2 MCTS with DPW** (Level 2). Online planning for the regime-switching MDP. DPWSolver from POMDPs.jl. Why online planning suits this problem: the expanded state space makes tabular methods intractable.

**3.3 DQN** (Level 2). Neural network architecture, replay buffer, ε-greedy exploration, target network. From scratch using Flux.jl. How the regime is encoded as a feature.

**3.4 QMDP** (Level 3). Solve the MDP per regime, weight Q-values by belief. From scratch. Discuss the QMDP approximation assumption (assumes full observability at next step) and when it breaks down — in particular, QMDP cannot exploit the fill signal's informational advantage because it does not plan through the belief update. This makes QMDP a useful lower bound on POMDP performance.

**3.5 POMCPOW** (Level 3). Online POMDP solver from POMDPs.jl. How it handles the continuous observation space (returns) and discrete action space. POMCPOW can in principle plan through the fill signal's informational content, making it the upper bound on learned POMDP performance.

### Section 4: Results (2 pages)

**4.1 Level 1 validation.** Value iteration policy matches analytical benchmarks. Policy visualization showing learned spread and hedge as functions of state variables. In Level 1, the market and agent have no regime uncertainty, so $V_{\text{market}} = V_{\text{believed}} = V_{\text{true-regime}}$ and the dual-filter architecture reduces to the standard model.

**4.2 Level 2: Regime-switching.** MCTS vs. DQN comparison. Both outperform constant-vol benchmarks. Show how the learned policy differs across regimes — wider spreads and more cautious hedging in the high-vol regime.

**4.3 Level 3: Hidden regime.** QMDP and POMCPOW results. Three nested performance comparisons, each isolating a distinct information effect:

- *Cost of partial observability:* Level 2 (regime known) minus Level 3 (regime hidden). This is the performance lost to not knowing the regime at all.
- *Value of the fill signal:* Level 3 with fill-augmented belief minus a returns-only agent (one whose belief update is identical to the market's). This isolates the value of the private observation channel.
- *Belief divergence dynamics:* Show a time series of the agent's belief vs. the market's belief, with the true regime overlaid. Annotate fill asymmetry events. Demonstrate that the agent's belief converges to the true regime faster than the market's — and that this speed advantage is what generates the fill signal's value.

**4.4 Solver comparison table.** Across all three levels, compare: cumulative P&L (mean ± std), Sharpe ratio, computation time, policy interpretability.

### Section 5: Discussion (0.5 pages)

Connect to course content: which DMU concepts proved most valuable for this application. The dual-filter POMDP architecture as a novel contribution: modeling the "observation environment" (the market) as itself a Bayesian agent with limited information formalizes the intuition that fill flow is private information, and gives the POMDP layer a rigorous economic justification rather than an ad hoc one. Limitations and future work.

---

## Key Figures (Both Papers)

1. **Environment schematic:** Flow diagram showing the full timestep sequence — observe → choose action → compute V_market from market belief → execute hedge → simulate fills against V_market → update market belief (returns only) → update agent belief (returns + fills) → spot moves → reward. The two parallel belief update arrows should be visually distinct, showing the information asymmetry.
2. **P&L distribution comparison:** Histograms for RL agent vs. analytical benchmarks (like AS Figure 2).
3. **Policy surface:** Heatmap of learned spread level and hedge ratio as functions of (Δ_net, moneyness) or (Δ_net, regime).
4. **Learned spread vs. GLF-T benchmark:** Overlay plot showing learned spread width vs. analytical optimal at each inventory level.
5. **Belief divergence plot:** Time series showing three lines — the true regime (step function), the agent's belief (returns + fills), and the market's belief (returns only) — over a single episode containing at least one regime switch. Annotate the moments where fill asymmetry causes the agent's belief to diverge from the market's. This figure appears in the DMU paper and optionally in the derivatives paper's appendix.
6. **Value of private information:** Bar chart or table decomposing performance into three layers: cost of partial observability (L2 − L3), value of fill signal (L3 with fills − L3 returns-only), and irreducible uncertainty floor.
7. **Transaction cost sensitivity:** Surface or heatmap showing how optimal policy and Sharpe ratio change with κ.
