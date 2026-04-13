# Paper Outlines

---

## Paper 1: Derivatives Course (Due April 17, 8–10 pages)

**Title:** Options Market Making Under Regime-Switching Volatility: A Simulation Study of Spread and Hedge Policy Interactions

**Emphasis:** Financial modeling, derivatives theory, market microstructure, benchmark comparison results.

**Scope note:** This paper covers Modules 1–8 only. RL solvers (value iteration, DQN, MCTS, QMDP, POMCPOW) are explicitly deferred to the companion DMU paper. The paper is self-contained — the simulation framework and benchmark results constitute a complete contribution without RL.

---

### Section 1: Introduction (1 page)

Motivate from the perspective of a real options market maker. The core tension: wider spreads earn more edge per fill but reduce order flow; frequent hedging reduces delta risk but incurs transaction costs proportional to trade size. Under stochastic volatility these tradeoffs are state-dependent — the optimal spread and hedge policy in a calm low-vol regime differs from the optimal policy in a turbulent high-vol regime.

Existing analytical solutions (Avellaneda-Stoikov 2008, Guéant et al. 2013, Whalley-Wilmott 1997) were derived for single-volatility environments. This paper builds a simulation framework that extends these benchmarks to a two-regime Markov-switching volatility environment calibrated to S&P 500 data, and evaluates how the spread and hedge components of each policy interact.

State three contributions:
1. Extension of the GLF-T spread formula and WW hedge formula to options via the dollar-gamma substitution γσ²τ → γ·Γ·S²·σ²·τ.
2. A simulation framework for options market making under regime-switching volatility with a transition-weighted market pricing model.
3. A four-policy factorial design isolating the independent and interactive contributions of the spread formula and hedge rule to P&L and Sharpe ratio.

---

### Section 2: Literature Review (1.5 pages)

**Market-making stream:** Ho & Stoll (1981) — inventory risk framework. Avellaneda & Stoikov (2008) — exponential fill intensity, optimal spread for stocks. Guéant, Lehalle & Fernandez-Tapia (2013) — rigorous control problem with bounded inventory, Proposition 3 closed form. Cartea, Jaimungal & Penalva (2015) — inventory penalty reward structure.

**Hedging-under-costs stream:** Black-Scholes (1973) — frictionless replication baseline. Leland (1985) — modified volatility under transaction costs. Davis, Panas & Zariphopoulou (1993) — option pricing as stochastic control under transaction costs. Whalley & Wilmott (1997) — asymptotic no-trade bandwidth, the key practical benchmark.

**Regime-switching stream:** Hamilton (1989) — Markov-switching model and filter. Hardy (2001) — calibrated RSLN-2 model for equity returns (our parameter source). Kim (1994) — dynamic linear models with Markov switching.

Gap to fill: no existing work combines market-making spread optimization with transaction-cost-aware hedging for options under regime-switching volatility, with all components (spread formula, hedge rule, market pricing) derived consistently within the same stochastic vol framework.

---

### Section 3: Model (2.5 pages)

**3.1 Spot price dynamics.**
Two-state regime-switching GBM with risk-neutral drift μ = r. Parameters from Hardy (2001): σ₁ = 12.1% (low-vol), σ₂ = 26.9% (high-vol), daily transition matrix [0.9982 0.0018; 0.0022 0.9978], stationary distribution π₁ ≈ 0.55, π₂ ≈ 0.45. Average regime duration ~500 trading days — regimes are highly persistent, switching rarely within a single episode. Risk-neutral drift prevents policies from learning directional biases.

**3.2 Market pricing and the fill model.**
The key modeling choice: what option price does the market use to decide whether to fill the agent's quotes?

We model the market as using a **transition-weighted Black-Scholes price**:

$$V_{\text{market}} = \sum_j P(\text{regime}_{t+1} = j \mid \text{regime}_t = i) \times V_{\text{BS}}(\sigma_j)$$

where $i$ is the current true regime. This is the one-step-ahead expected fair value given perfect regime knowledge. It is marginally different from $V_{\text{BS}}(\sigma_i)$ (the current-regime price) because it accounts for the small probability of switching volatility by the next step. This is more intellectually defensible than treating the market as clairvoyant about instantaneous vol — real market consensus aggregates participants' estimates of near-future pricing, not just current-moment pricing.

Fill probabilities follow Avellaneda-Stoikov (2008): $\lambda(\delta) = A e^{-k\delta}$ where $\delta$ is the distance between the agent's quote and V_market. The agent quotes around its own believed fair value; when the agent uses the transition-weighted oracle σ (as in these benchmarks), quotes are well-centered and fills are approximately symmetric.

**3.3 Portfolio and reward.**
The agent holds an inventory of options and a hedge position in the underlying. Portfolio Greeks (Δ, Γ, ν, Θ) are computed via belief-weighted Black-Scholes. The reward at each step is:

$$r_t = \Delta\text{PnL}_t - \phi \cdot \Delta_{\text{net},t}^2$$

where the four P&L components are: (1) mark-to-market on options, (2) spread capture from fills, (3) hedge P&L on underlying shares, (4) hedge transaction cost $\kappa \cdot |\text{shares traded}| \cdot S$. The quadratic delta penalty $\phi \cdot \Delta_{\text{net}}^2$ is adapted from Cartea et al. (2015) and is proportional to P&L variance under Gaussian returns (CARA equivalence).

**3.4 Spread action space.**
Six discrete half-spread levels: $[0.05, 0.10, 0.20, 0.40, 0.80, 1.60]$ dollars. At each step the agent chooses a level; `compute_quotes` centers bid and ask symmetrically around V_believed at the chosen half-spread.

**3.5 Hedge action space.**
Fourteen discrete absolute net-delta targets: $\{:\text{no\_trade}, -0.30, -0.25, \ldots, 0.25, 0.30\}$. The agent picks a destination in net-delta space; `execute_hedge!` trades the difference between the target and current net_Δ, charging κ·|shares|·S. `:no_trade` is zero cost and corresponds to the WW "inside the band" decision.

Absolute targets were chosen over fractional targets because net_Δ is directly observable while Δ_options in isolation is not, and the transaction cost of each action is a direct function of the observable state.

---

### Section 4: Analytical Benchmarks (1 page)

Four policies evaluated in a factorial design — two spread choices × two hedge choices — to isolate each component's contribution.

**Spread policies:**

*GLF-T (primary):* From Guéant, Lehalle & Fernandez-Tapia (2013), adapted for options via dollar-gamma substitution:

$$\delta^*_{\text{GLF-T}} = \gamma \cdot \Gamma S^2 \sigma^2 \tau + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right)$$

The first term is the inventory risk premium — it scales with dollar-gamma $\Gamma S^2 \sigma^2$, not raw volatility, because the P&L variance of an options position is driven by gamma rather than vol directly. This produces spreads that widen near-the-money and near expiry — exactly the behavior real options market makers exhibit. The dollar-gamma substitution is a stated contribution.

*Naive (baseline):* Fixed half-spread of $0.10 regardless of market conditions. Represents a market maker who sets spreads by convention rather than formula.

**Hedge policies:**

*Whalley-Wilmott (primary):* No-trade band with half-bandwidth:

$$H = \left(\frac{3\kappa}{2\phi} \cdot \Gamma^2 S^2 \sigma^2\right)^{1/3} \cdot \Delta t^{1/3}$$

When $|\text{net\_Δ}| \leq H$: take the `:no_trade` action (zero cost). When outside: trade to the band edge. Again the dollar-gamma adaptation is required — the original WW formula uses stock-level volatility while we use option dollar-gamma.

*Naive (baseline):* Always target net_Δ = 0. Rebalances fully every step regardless of transaction cost.

**Oracle σ for all policies:** All benchmark policies receive the transition-row-weighted variance-equivalent effective volatility:

$$\sigma_{\text{eff}} = \sqrt{\sum_j P(\text{regime}_{t+1} = j \mid \text{regime}_t = i) \cdot \sigma_j^2}$$

This is the natural single-σ input consistent with the two-regime world — it gives the benchmarks the same information the market uses, ensuring that any performance differences are due to policy quality rather than information asymmetry.

---

### Section 5: Results (3 pages)

**5.1 Spread behavior (Figure 2).**
GLF-T spread width rises from ~$0.40 at full expiry (63 days) to ~$0.64 at peak (~45 days) then declines near expiry as tau → 0 reduces the inventory risk term. The peak at 45 days (not at 1 day) reflects the gamma term: ATM gamma is highest near expiry but the τ multiplier in the GLF-T formula reduces spread as tau → 0, producing the observed hump shape. Under regime-switching, the peak is slightly lower (~$0.60) because the effective σ from transition-weighting is slightly lower than the constant-vol σ = 0.20 in the low-vol regime. Naive spread is flat at $0.10 by construction.

**5.2 Hedge behavior (Figure 3).**
WW policies hedge approximately 45% of timesteps versus 100% for Naive hedge. This is the WW no-trade band functioning as designed — roughly half the time the portfolio delta is within the band [−H, H] and trading is suboptimal given transaction costs. Naive+WW hedges slightly more frequently (62%) than GLF-T+WW (45%) because the tight spread causes faster inventory accumulation, pushing delta outside the band more often.

Mean |net Δ| is lowest for GLF-T+Naive (0.054) despite 100% hedge frequency. This is because the wide GLF-T spread produces few fills and therefore slow inventory accumulation — there is rarely much delta to hedge, so each rebalancing trade is small. WW with a naive spread produces the highest |net Δ| (0.220) because the tight spread floods inventory while WW allows delta to accumulate within its no-trade band.

**5.3 P&L distributions (Figure 1) and summary table.**

*Table 1* [include full table from results].

Three findings:

*Finding 1 — Spread formula dominates variance reduction.* GLF-T cuts Std P&L roughly in half versus Naive spread (6.06 vs ~10–12). Hedging choice has minimal effect on variance when spread is already wide. The wide spread reduces fill frequency and inventory accumulation, which is the primary source of P&L variance in this environment.

*Finding 2 — WW's value is cost savings, not delta management.* GLF-T+WW saves $1.04/episode in hedge costs versus GLF-T+Naive ($3.73 vs $4.77) while producing essentially the same mean P&L and variance. For Naive spreads, WW saves $3.57/episode ($14.45 vs $18.02). However, naive hedge costs ($18.02/episode) exceed total mean P&L ($11.39), meaning the strategy would be unprofitable if not for spread capture income. WW is an important cost optimization but does not rescue a fundamentally misspecified spread policy.

*Finding 3 — Regime switching amplifies Naive policy variance disproportionately.* Std P&L for GLF-T+WW increases 6.6% under regime switching (6.06 → 6.46). For Naive+WW it increases 20.3% (11.93 → 14.35). This asymmetry arises because GLF-T's spread formula adapts to the current effective σ — when regime-switching changes the oracle σ, GLF-T adjusts spread width accordingly. Naive spreads are fixed at $0.10 regardless of vol regime, so they absorb the full variance impact of regime switches. This result directly motivates the DMU paper's RL extension: a policy that can learn to adapt both spread and hedge jointly to regime changes should recover the variance penalty.

**5.4 Cumulative P&L traces (Figure 4).**
Single-episode traces under constant vol and Hardy regime-switching show the staircase structure characteristic of spread capture — P&L accumulates in discrete jumps at fill events, interrupted by hedge costs. GLF-T policies show smoother, more monotone accumulation; Naive policies show wider swings. High-vol regime periods (shaded) do not always correlate with drawdowns because regime-switching vol also widens GLF-T spreads, increasing per-fill revenue to partially offset increased delta risk.

---

### Section 6: Discussion and Future Work (0.5 pages)

**Main takeaway:** The spread formula is the primary determinant of P&L quality under stochastic volatility. GLF-T's dollar-gamma adaptation correctly widens spreads when gamma is high (near expiry, ATM), which both increases per-fill revenue and reduces inventory accumulation. WW adds meaningful value through transaction cost savings but cannot rescue a misspecified spread policy.

**Limitations:** Stylized simulation, single strike, no empirical calibration beyond Hardy (2001) parameters, discrete action space may underperform continuous policies near WW band boundaries.

**Future work:** The natural extension is a learned policy that jointly optimizes spread and hedge through simulation — an MDP/POMDP formulation solved via reinforcement learning. Such a policy would be evaluated against these analytical benchmarks and its advantage would be precisely quantifiable through the factorial decomposition introduced here. This is the subject of the companion DMU paper.

---

## Paper 2: DMU Course (Due April 30, 4–8 pages)

**Title:** Joint Spread and Hedge Optimization for Options Market Making: From MDPs to POMDPs Under Hidden Volatility Regimes

**Emphasis:** MDP/POMDP formulation, solver comparison, cost of partial observability, fill asymmetry as private information signal.

**Note:** This paper presupposes the derivatives paper's financial setup. It focuses narrowly on the algorithmic contributions and DMU course connections. Results from Modules 9–13 are the primary content.

---

### Section 1: Introduction (0.5 pages)

Options market making is a natural DMU application: sequential decision-making under uncertainty (MDP), partial observability of volatility regime (POMDP), exploration-exploitation (RL training), and Bayesian state estimation (Hamilton filter). Briefly state the three levels of complexity and preview the two key measurements: cost of partial observability and value of the fill signal as private information.

---

### Section 2: Problem Formulation (1.5 pages)

**2.1 MDP formulation.** State space: (S, τ, net_Δ, net_Γ, net_ν, net_Θ, regime_belief). Action space: 6 spread × 14 hedge = 84 actions. Transition dynamics: regime-switching GBM + AS fill model. Reward: pnl - φ·Δ_net². Formal notation matching Kochenderfer's textbook.

Reference the four analytical benchmark results from the derivatives paper as the performance baseline against which RL policies are compared.

**2.2 POMDP extension.** Hidden state: volatility regime. Observations: log return (public signal of regime magnitude) and fill outcome (private signal — the market cannot observe the agent's fills).

The dual-filter architecture: the market maintains a belief updated on log returns only (determines V_market). The agent maintains a separate belief updated on both log returns and fill outcomes. The two filters run in parallel, diverging when fill asymmetry provides information that the return signal has not yet delivered. This makes fill asymmetry formally informative: because V_market is belief-weighted (not true-regime), the agent's fill-augmented belief can advance ahead of the market's, giving a quantifiable private information advantage.

---

### Section 3: Solvers (1.5 pages)

**3.1 Value iteration (Level 1).** Tabular VI over discretized state space. Bellman backup, convergence. Policy visualization: learned spread and hedge as functions of (net_Δ, τ, moneyness).

**3.2 MCTS with DPW (Level 2).** DPWSolver from POMDPs.jl. Why online planning: expanded regime-switching state space makes tabular methods intractable.

**3.3 DQN (Level 2).** 3-layer MLP, replay buffer, ε-greedy, target network. From scratch with Flux.jl. Regime index encoded as one-hot feature.

**3.4 QMDP (Level 3).** Solve MDP per regime via VI, weight Q-values by Hamilton filter belief: a* = argmax_a Σⱼ belief_j · Q_j(s,a). QMDP approximation assumption: treats next state as fully observable. Cannot exploit fill signal's informational content — useful as lower bound on POMDP performance.

**3.5 POMCPOW (Level 3).** Online POMDP solver from POMDPs.jl. Handles continuous observation space (log returns) with progressive widening. Can in principle plan through the fill signal update — upper bound on learned POMDP performance.

---

### Section 4: Results (2 pages)

**4.1 Level 1 validation.** VI policy matches GLF-T spread and WW hedge in the constant-vol case, confirming the simulation is correctly implemented.

**4.2 Level 2: Regime-switching.** MCTS vs DQN comparison against GLF-T+WW and GLF-T+Naive benchmarks. Show how learned spread policy adapts across regimes — wider spreads and more conservative hedging in high-vol regime.

**4.3 Level 3: Hidden regime.** Three nested comparisons:
- *Cost of partial observability:* Level 2 (known regime) minus Level 3 (hidden regime). Performance lost to not observing the regime.
- *Value of fill signal:* Level 3 (returns + fills) minus Level 3 (returns only). Isolates the private information channel.
- *Belief convergence:* Time series of agent belief vs market belief vs true regime over an episode with a regime switch. Show agent converges faster than market due to fill asymmetry.

**4.4 Solver comparison table.** Mean P&L ± std, Sharpe, compute time, interpretability across all levels and solvers.

---

### Section 5: Discussion (0.5 pages)

Connect to DMU course content: MDP (Levels 1–2), POMDP (Level 3), value iteration, online tree search, DQN, Hamilton filter as Bayesian state estimator, fill asymmetry as private observation. The dual-filter architecture as a contribution: modeling the market as a Bayesian agent with limited information gives the POMDP layer a rigorous economic justification rather than treating partial observability as a purely algorithmic challenge.

---

## Figures

### Derivatives Paper Figures (complete)

1. **Fig 1 — P&L distributions:** Histograms of episode P&L for all 4 policies under constant vol and Hardy regime-switching. ✓ Complete.
2. **Fig 2 — Spread vs τ:** Mean GLF-T half-spread as a function of days to expiry. Shows hump-shaped pattern from dollar-gamma term. ✓ Complete.
3. **Fig 3 — Hedge behavior:** Bar charts of hedge frequency and mean |net Δ| across all 4 policies. ✓ Complete.
4. **Fig 4a — Cumulative P&L, constant vol:** Single episode trace. ✓ Complete.
5. **Fig 4b — Cumulative P&L, Hardy:** Single episode trace with regime shading. ✓ Complete (legend vspan bug is cosmetic, not blocking).

### DMU Paper Figures (pending — April 18–30)

6. **Environment schematic:** Timestep flow diagram showing dual belief update architecture.
7. **Policy surface:** Heatmap of learned spread and hedge as functions of (net_Δ, moneyness) under Level 1 VI.
8. **Belief divergence plot:** Agent belief vs market belief vs true regime over an episode with a switch.
9. **Value of private information:** Bar chart decomposing L2 − L3 gap and fill-signal value.
10. **Solver comparison table:** Full cross-level results table.
