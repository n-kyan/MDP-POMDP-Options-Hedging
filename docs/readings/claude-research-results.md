# Six foundational pillars for an options market-making agent

This report assembles the key papers, formulas, and implementation details for building an RL-based options market-making agent that jointly optimizes bid-ask spread width and delta-hedge ratio under regime-switching volatility. **No existing paper combines all four core elements of this project** — RL spread optimization, RL hedge ratio learning, options as the asset class, and regime-switching volatility — establishing clear novelty. Below, each of the six research topics is presented with precise citations, implementable formulas, and their exact connection to the project.

---

## Topic 1: Regime-switching volatility calibrated to the S&P 500

The canonical calibration is **Hardy (2001)**, not Hamilton (1989). Hamilton's foundational paper estimates regime-switching on quarterly GNP growth, whereas Hardy provides a clean 2-regime log-normal model fit directly to S&P 500 total returns.

**Primary paper:**
Hardy, M.R. (2001). "A Regime-Switching Model of Long-Term Stock Returns." *North American Actuarial Journal*, 5(2), 41–53.

**Model (RSLN-2):** Monthly log-returns follow Y_t | ρ_t ~ N(μ_{ρ_t}, σ²_{ρ_t}) where ρ_t ∈ {1, 2} is a Markov chain. Six parameters estimated by MLE on S&P 500 total return index, January 1956 – December 1999:

| Parameter | Regime 1 (Bull) | Regime 2 (Bear) |
|-----------|-----------------|-----------------|
| μ (monthly) | **0.0114** | **−0.0066** |
| σ (monthly) | **0.0350** | **0.0778** |
| σ annualized (×√12) | ≈ 12.1% | ≈ 26.9% |

**Transition probability matrix:**

|  | → Regime 1 | → Regime 2 |
|--|-----------|-----------|
| From Regime 1 | **0.9626** | **0.0374** |
| From Regime 2 | **0.0462** | **0.9538** |

**Stationary distribution:** π₁ ≈ 0.553 (bull), π₂ ≈ 0.447 (bear). Expected durations are **26.7 months** (bull) and **21.6 months** (bear). The model was validated against GARCH, MARCH, and other alternatives, winning by BIC. Hardy's Table 1 (Section 4) contains all parameter estimates.

For daily conversion: σ_daily = σ_monthly / √21 and off-diagonal transition probabilities scale as p_daily ≈ 1 − (1 − p_monthly)^{1/21}, yielding p₁₂ ≈ 0.0018 and p₂₁ ≈ 0.0022 per trading day.

**Connection to the project:** These calibrated volatility levels and transition matrix parameterize the regime-switching environment for the MDP/POMDP simulator.

**Survey reference:** Ang, A. and Timmermann, A. (2012). "Regime Changes and Financial Markets." *Annual Review of Financial Economics*, 4, 313–337. This review confirms that regimes are identified primarily by **volatility** rather than mean returns, with both persistence parameters exceeding 0.90.

---

## Topic 2: Three analytical benchmarks for hedging under transaction costs

These three papers form a hierarchy — from a simple heuristic (Leland) to a closed-form approximation (Whalley-Wilmott) to the exact theoretical optimum (Davis-Panas-Zariphopoulou) — against which the RL agent's hedging policy can be benchmarked.

### Leland (1985): Modified volatility

Leland, H.E. (1985). "Option Pricing and Replication with Transactions Costs." *The Journal of Finance*, 40(5), 1283–1301.

**Key formula** (p. 1286):

$$\hat{\sigma}^2 = \sigma^2 \left(1 + \sqrt{\frac{2}{\pi}} \cdot \frac{\kappa}{\sigma \sqrt{\Delta t}}\right)$$

where σ is the true volatility, κ is the round-trip proportional transaction cost rate, and Δt is the rebalancing interval (e.g., 1/252 for daily). The term √(2/π) ≈ 0.7979 comes from E[|Z|] for standard normal Z. Implementation: replace σ with σ̂ in Black-Scholes for both pricing and delta computation. Note that Kabanov & Safarian (1997, *Finance and Stochastics*) proved the convergence claim is flawed for fixed κ, but the formula remains a practical heuristic.

**Connection to the project:** Provides the simplest cost-adjusted hedging baseline — the RL agent should outperform Leland's modified-delta strategy.

### Whalley & Wilmott (1997): No-trade bandwidth

Whalley, A.E. and Wilmott, P. (1997). "An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs." *Mathematical Finance*, 7(3), 307–324.

**Key formula** (Section 3, proportional costs):

$$H = \left(\frac{3}{2} \cdot \frac{\kappa \, S \, e^{-r(T-t)} \, \Gamma^2}{\gamma}\right)^{1/3}$$

The optimal policy is: **do not trade** if the current hedge position y satisfies Δ_BS − H ≤ y ≤ Δ_BS + H. When y exits this band, trade minimally to return to the nearest boundary. Here κ is the proportional cost rate, S the stock price, Γ = ∂²V/∂S² the Black-Scholes gamma, and γ the CARA risk-aversion parameter. The bandwidth scales as **κ^{1/3}**, meaning even small costs create meaningful no-trade regions. Near the money (high Γ) the band widens — hedge less often where gamma is large.

**Connection to the project:** The Whalley-Wilmott bandwidth defines the analytically optimal hedge-timing policy; the RL agent's learned hedge threshold should approximate or improve upon this formula.

### Davis, Panas & Zariphopoulou (1993): Theoretical foundation

Davis, M.H.A., Panas, V.G., and Zariphopoulou, T. (1993). "European Option Pricing with Transaction Costs." *SIAM Journal on Control and Optimization*, 31(2), 470–493.

This paper rigorously proves via singular stochastic control that the **optimal hedging policy under proportional transaction costs is a no-trade band** — hold the current position unless exposure leaves a regime-dependent boundary. The utility-indifference option price is defined through a 3D free boundary problem with no closed-form solution. Whalley-Wilmott is the asymptotic (small-cost) approximation to this exact solution, losing the asymmetry between long and short gamma and the volatility shift. The DPZ framework establishes that under CARA utility U(x) = 1 − exp(−γx) with proportional costs, the band structure is theoretically optimal.

**Connection to the project:** DPZ proves that the "hedge only outside a threshold" policy structure is exactly optimal — the RL agent's policy should converge to this structure, validating the approach.

---

## Topic 3: Guéant, Lehalle & Fernandez-Tapia extend Avellaneda-Stoikov with inventory bounds

**Citation:** Guéant, O., Lehalle, C.-A., and Fernandez-Tapia, J. (2013). "Dealing with the inventory risk: a solution to the market making problem." *Mathematics and Financial Economics*, 7(4), 477–507. Also arXiv:1105.3115.

This paper resolves a key limitation of Avellaneda-Stoikov by imposing **discrete bounded inventory** q ∈ {−Q, …, Q} and deriving exact optimal quotes via a system of linear ODEs rather than asymptotic expansions.

**Optimal bid/ask depths** (Theorem 1):

$$\delta^*_b(t, q) = \frac{1}{k}\ln\frac{v_q(t)}{v_{q+1}(t)} + \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right), \quad q \neq Q$$

$$\delta^*_a(t, q) = \frac{1}{k}\ln\frac{v_q(t)}{v_{q-1}(t)} + \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right), \quad q \neq -Q$$

At inventory limits (q = Q or q = −Q), the corresponding quote is withdrawn. The functions v_q(t) solve a **tridiagonal linear ODE system**:

$$\dot{v}_q = \alpha q^2 \, v_q - \eta(v_{q-1} + v_{q+1}), \quad |q| < Q$$

with boundary modifications at ±Q and terminal condition v_q(T) = 1 for all q. The parameters are **α = (k/2)γσ²** and **η = A(1 + γ/k)^{-(1+k/γ)}**. In matrix form: v(t) = exp(−M(T−t)) · **1** where M is a (2Q+1) × (2Q+1) tridiagonal matrix.

**Closed-form asymptotic approximation** (Proposition 3, valid for large T−t):

$$\delta^*_{b,\infty}(q) \approx \frac{1}{\gamma}\ln\!\left(1 + \frac{\gamma}{k}\right) + \frac{2q+1}{2}\sqrt{\frac{\sigma^2 \gamma}{2kA}\left(1 + \frac{\gamma}{k}\right)^{1+k/\gamma}}$$

$$\delta^*_{a,\infty}(q) \approx \frac{1}{\gamma}\ln\!\left(1 + \frac{\gamma}{k}\right) - \frac{2q-1}{2}\sqrt{\frac{\sigma^2 \gamma}{2kA}\left(1 + \frac{\gamma}{k}\right)^{1+k/\gamma}}$$

**How it reduces to AS:** The Avellaneda-Stoikov formulas are Taylor expansions of the GLF-T solutions for t close to T (short horizon). AS gives δ ∝ γσ²(T−t) which grows linearly and diverges — the GLF-T asymptotic formulas are bounded and more realistic for finite horizons. The paper provides a rigorous verification theorem that AS itself lacks.

**Connection to the project:** GLF-T provides a strictly better analytical benchmark than raw AS for finite inventory, with implementable closed-form quotes that the RL agent must outperform.

---

## Topic 4: The right risk measure connects CARA utility to quadratic PnL penalties

**Primary reference:** Cartea, Á., Jaimungal, S., and Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press. Chapter 10, §10.3–10.4.

Cartea, Jaimungal & Penalva use a **mean-variance formulation with running inventory penalty** as their primary objective:

$$\sup \; \mathbb{E}\!\left[X_T + q_T S_T - \varphi \int_0^T q_t^2 \, dt - \alpha \, \ell(q_T)\right]$$

where φ ≥ 0 is the risk aversion parameter and the running penalty φq² proxies for variance of inventory value via Itô isometry. This is preferred over CARA utility because it yields more tractable solutions and generalizes cleanly to multi-asset settings. The running inventory penalty was shown by Cartea & Jaimungal (2015, "Risk Metrics and Fine Tuning of High-Frequency Trading Strategies," *Mathematical Finance*, 25, 576–611) to be "much more effective than the terminal inventory constraint at controlling the trader profit and loss distribution."

### CARA-to-mean-variance equivalence is exact under Gaussian returns

For CARA utility u(x) = −exp(−γx) with x ~ N(μ, σ²), the moment-generating function gives:

$$\mathbb{E}[-e^{-\gamma x}] = -\exp\!\left(-\gamma \mu + \frac{\gamma^2 \sigma^2}{2}\right)$$

Maximizing this is **exactly equivalent** to maximizing μ − (γ/2)σ², i.e., E[x] − (γ/2)Var[x]. This is not merely a Taylor approximation — it is exact for Gaussian distributions. For non-Gaussian returns, the equivalence holds as a second-order approximation for "small" risks. Setting λ = γ/2 in the project's reward r = pnl − λ·pnl² recovers the CARA certainty equivalent under normality.

**Critical observation about the project's reward function:** The formulation r = pnl − λ·pnl² is not pure mean-variance because E[pnl − λ·pnl²] = E[pnl] − λ(Var[pnl] + E[pnl]²). The squared-mean term means this penalizes consistently profitable strategies. CJP's formulation penalizes q² (the *source* of risk) rather than pnl² (its *consequence*), which is arguably better targeted. A recommended alternative is r = pnl − φ·q², matching CJP's running inventory penalty directly.

**CVaR for market making** has not been used in any published spread-optimization paper, though Ni, Liu & Lai (2024, ICML) provide a general CVaR-RL framework (CVaR-RF-UCRL) applicable to any MDP. The **Differential Sharpe Ratio** introduced by Moody & Saffell (2001, *NeurIPS*) is an alternative per-step RL reward but is untested for market making.

**Connection to the project:** The quadratic penalty is theoretically grounded through its equivalence to CARA under Gaussian assumptions, but switching to a running inventory penalty r = pnl − φ·q² would better match CJP's recommended formulation.

---

## Topic 5: The novelty claim holds — no paper jointly learns spread and hedge for options

A systematic review of the RL hedging and market-making literature reveals a clear gap at the intersection of spread optimization and hedge-ratio learning for options.

| Paper | Spread optimization | Hedge learning | Asset class | Vol model |
|-------|:------------------:|:--------------:|:-----------:|:---------:|
| Buehler et al. (2019), *Quantitative Finance* 19(8) | ✗ | ✓ | Options | Heston |
| Fang & Xu (2023), arXiv:2307.01814 | ✓ | ✗ (perfect δ) | Options | GBM |
| Shi, Tang & Zhou (2024), ICAIF '24 | ✓ | ✓ | FX spot | — |
| Sadighian (2019/2020), arXiv:1911.08647 | ✓ | ✗ | Crypto spot | — |
| Fathi & Hientzsch (2023), arXiv:2302.07996 | ✗ | ✓ | Options | BSM |
| Cao, Chen, Hull & Poulos (2021), *J. Fin. Data Sci.* | ✗ | ✓ | Options | SABR |

**Buehler et al. (2019)** — "Deep Hedging" — trains neural networks end-to-end to minimize convex risk measures of terminal P&L. Despite being described as "deep RL," the optimization uses policy gradient over full trajectories, closer to stochastic optimal control than model-free RL. It handles hedging only with **no spread setting**. Demonstrated on S&P 500 options under Heston dynamics.

**Fang & Xu (2023)** — "Option Market Making via Reinforcement Learning" — is the closest paper. It uses RL to optimize bid-ask spreads for options across strikes and maturities. However, it **assumes perfect analytical delta hedging** — the hedge ratio is computed from Black-Scholes and applied deterministically, not learned. The volatility model is standard GBM with no regime-switching.

**Shi, Tang & Zhou (2024)** — the only paper that jointly optimizes quoting and hedging via deep RL, but applies to **FX spot** markets, not options. Uses partially monotonic networks to encode prior knowledge.

**Pickard & Lawryshyn (2023)** — "Deep Reinforcement Learning for Dynamic Stock Option Hedging: A Review" (*Mathematics*, 11(24), 4943) — a survey of 17 DRL hedging studies finding DDPG as the dominant algorithm, mean-variance as the dominant reward, and GBM as the dominant data process. **No paper in their survey combines spread optimization with hedging.**

**The novelty is threefold:** (1) No paper jointly *learns* both spread width and hedge ratio for options — Fang & Xu hardcode the hedge. (2) No RL hedging or market-making paper uses regime-switching volatility. (3) The full four-way combination (joint learning + options + regime-switching + POMDP regime inference) is completely unprecedented.

**Connection to the project:** This gap establishes the project's primary contribution to the literature.

---

## Topic 6: The Hamilton filter gives an O(K²) recursive update for regime beliefs

**Primary reference:** Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357–384. The clearest general presentation is in Hamilton (1994), *Time Series Analysis*, Chapter 22, Eqs. (22.4.1)–(22.4.5).

The filter maintains a K-dimensional belief vector ξ̂_{t|t} = Pr(S_t = j | y_1, …, y_t) and updates it recursively in four steps:

**Step 1 — Predict:** ξ̂_{t|t−1} = P^⊤ · ξ̂_{t−1|t−1}

**Step 2 — Likelihood:** η_{jt} = (2πσ²_j)^{−1/2} exp(−(r_t − μ_j)² / (2σ²_j)) for each regime j

**Step 3 — Evidence:** L_t = ξ̂^⊤_{t|t−1} · η_t

**Step 4 — Update:** ξ̂_{t|t} = (ξ̂_{t|t−1} ⊙ η_t) / L_t

where ⊙ denotes element-wise multiplication. For K = 3 regimes, this requires **9 multiplications** (the matrix-vector product), 3 Gaussian density evaluations, and a normalization — trivial computational cost. Initialize with the ergodic distribution or uniform 1/K.

The **Kim (1994) smoother** — from Kim, C.-J. (1994), "Dynamic Linear Models with Markov-Switching," *Journal of Econometrics*, 60(1–2), 1–22 — adds backward-pass smoothed probabilities:

$$\xi_{t|T} = \xi_{t|t} \odot \left[P^⊤ \cdot (\xi_{t+1|T} \oslash \xi_{t+1|t})\right]$$

This is useful for offline calibration but not needed for online POMDP inference. The definitive implementation reference is Kim, C.-J. and Nelson, C.R. (1999), *State-Space Models with Regime Switching*, MIT Press.

### Order flow signals for regime detection go beyond returns

**Tsaknaki, Lillo & Mazzarisi (2024)** — "Online Learning of Order Flow and Market Impact with Bayesian Change-Point Detection Methods," *Quantitative Finance* (published online 2024; arXiv:2307.02375) — apply Bayesian Online Change-Point Detection to NASDAQ order flow data. They detect regimes in aggregated buy/sell trade sequences using a novel Markov Bayesian Online Changepoint (MBOC) model with score-driven parameters. Their key finding: **order flow long memory is largely explained by regime switching**, not within-regime autocorrelation. Regimes correspond to metaorder execution periods, with mean durations of 7–15 trading minutes.

A complementary paper on **inter-trade duration regime switching** (*Economic Modelling*, 2023) shows that LOB factors — spread, depth, order book imbalance — drive regime transition probabilities, linking regimes to HFT liquidity provision behavior.

**Connection to the project:** The Hamilton filter provides the POMDP belief-update step at negligible computational cost (O(K²) per tick). The Tsaknaki et al. result suggests augmenting return-based regime detection with fill-rate observations — if the agent's own fill rates change sharply, this carries regime information beyond what returns alone reveal.

---

## Conclusion

The six topics assemble into a coherent architecture. Hardy (2001) parameterizes the regime-switching environment, with the Hamilton filter providing online belief updates for the POMDP layer at O(K²) cost. Guéant-Lehalle-Fernandez-Tapia (2013) supplies a strictly better analytical benchmark than raw Avellaneda-Stoikov for spread-setting, while Whalley-Wilmott (1997) and Leland (1985) provide implementable closed-form hedge-timing baselines. The reward function r = pnl − λ·pnl² is theoretically grounded through its CARA equivalence under Gaussian returns, though Cartea-Jaimungal-Penalva's running inventory penalty φq² may be more effective. Most importantly, the literature review confirms that **no existing paper jointly learns spread width and hedge ratio for options under regime-switching volatility** — Fang & Xu (2023) hardcode the hedge, Shi et al. (2024) address FX rather than options, and no RL paper uses regime-switching dynamics. The project occupies a genuine gap at the intersection of four independently explored directions.