# Options Market Making Under Regime-Switching Volatility: An MDP Approach with Analytical Benchmarks

**Kyan Nelson**
ASEN 5264  Decision Making Under Uncertainty, Spring 2026

---

## 1. Introduction

Options market making (OMM) is a sequential decision problem under uncertainty. A market maker (MM) continuously posts a bid and an ask around an estimated fair value $\hat{V}_t$, earning the spread when trades execute against their quotes. The core tension is immediate: tighter spreads attract more order flow and increase fill frequency, but they reduce per-trade revenue and expose the MM to inventory accumulation. Wider spreads generate more revenue per fill but reduce flow and risk being consistently undercut by competitors.

What makes OMM genuinely difficult is the structure of the uncertainty the MM operates under. The two quantities most critical to the quoting and hedging decisions are the fair value of the option $V^*_t$ and the true instantaneous volatility $\sigma^*_t$. Neither is directly observable; both must be estimated from market data. Every pricing and risk computation the MM performs is downstream of these latent quantities, which means every quote is placed under epistemic uncertainty about the very inputs that determine whether that quote is profitable.

The analytical literature on market making has developed strong models for each sub-problem in isolation: Black-Scholes-Merton (BSM) [1] for option valuation, Avellaneda-Stoikov (AS) and Guéant-Lehalle-Fernandez-Tapia (GLFT) [2] for optimal spread-setting under inventory risk, Glosten-Milgrom [3] for adverse selection pricing, and Whalley-Wilmott [4] for hedging under transaction costs. These models each assume a specific, known volatility. Hardy (2001) [5] demonstrated that historical equity volatility is better described by a two-regime Markov chain, a structure that is empirically important but for which no single closed-form market-making policy integrates all four of the above decisions simultaneously.

This paper formulates OMM as a fully observable Markov Decision Process (MDP) in which the agent is given oracle access to the current volatility regime and the Hardy transition matrix, and thus can compute the belief-weighted BSM fair value exactly. The agent selects spread and hedge actions using Monte Carlo Tree Search with Double Progressive Widening (MCTS-DPW). The MDP policy is then benchmarked against four analytical solutions formed by combining the GLFT and Naive spread formulas with the Whalley-Wilmott and Naive hedging rules to compare the efficacy of a reinforcement learning solution relative to domain-derived closed-form policies under identical information. This comparison establishes a concrete ceiling for the analytical benchmarks and a floor for any future partially observable extension.

---

## 2. Background and Related Work

**Options pricing and the BSM framework.** Black, Scholes, and Merton [1] derived the canonical closed-form pricing formula for European options under the assumptions of continuous trading, log-normally distributed prices, and constant volatility. Despite these assumptions being empirically violated, BSM remains the industry-standard tool for computing option fair values and Greeks, the first-order sensitivities ($\Delta$, $\Gamma$) that characterize how an option position changes with the underlying. In this work, BSM is used as the agent's pricing tool: the agent uses BSM under the belief-weighted volatility $\hat{\sigma}_t$ to compute portfolio Greeks that enter the reward function and to anchor its fair value estimate $\hat{V}_t$.

**Optimal quoting under inventory risk.** Avellaneda and Stoikov [2a] formulated the market maker's spread-setting problem as a stochastic control problem, deriving a reservation price that adjusts for inventory and an optimal spread as a function of inventory, time horizon, and risk aversion. Guéant, Lehalle, and Fernandez-Tapia [2b] extended this to a tractable closed-form solution (GLFT) with exponential fill intensity, giving the MM explicit formulas for bid and ask offsets that scale with portfolio gamma, variance, and time to expiry. GLFT is used as the primary quoting benchmark in this paper.

**Adverse selection and informed order flow.** Glosten and Milgrom [3] modeled the bid-ask spread as an equilibrium outcome of a game between market makers and traders who are either informed or liquidity-motivated. A central insight is that fill events are Bayesian signals: an ask fill is evidence that the buyer believes the true value exceeds the ask, and the MM should update their fair value estimate accordingly. The current implementation delegates adverse selection protection to spread width rather than maintaining a separate flow-adjusted fair value; the Glosten-Milgrom framework points toward a natural future extension.

**Hedging under transaction costs.** Whalley and Wilmott [4] solved the problem of delta hedging with proportional transaction costs, showing that the optimal strategy is a no-trade band around the target delta. Hedging only when the portfolio delta drifts outside the band minimizes expected costs while bounding delta exposure. This is used as the hedging benchmark and directly informs the hedge action space: the agent's hedge target is constrained to reduce $|\hat{\Delta}_{P,t}|$ without flipping sign.

**Regime-switching volatility.** Hardy [5] demonstrated that a two-state regime-switching model fits historical S&P 500 returns better than GARCH by the Schwarz-Bayes Criterion, estimating regime volatilities of approximately 12.1% and 26.9% annualized with daily transition probabilities that produce regimes lasting on the order of months. The simulation environment in this paper uses Hardy's estimated parameters directly.

---

## 3. Problem Formulation

The OMM problem is formulated as a fully observable MDP $\langle \mathcal{S}, \mathcal{A}, T, R, \gamma \rangle$. In this formulation the agent receives oracle information: it observes the true current regime index and has access to the Hardy transition matrix $\mathbf{P}$. This is the same information used by the analytical benchmarks, so any performance gap reflects policy quality rather than information asymmetry.

**State Space $\mathcal{S}$**

The state of the world at each timestep is:
$$
s_t = (S_t,\ \text{regime}_t,\ q_t,\ \tau_t,\ q^h_t,\ \text{cash}_t,\ K_t)
$$
where $S_t \in \mathbb{R}_{>0}$ is the underlying spot price, $\text{regime}_t \in \{1, 2\}$ is the current volatility regime index (observable to the oracle agent), $q_t \in \mathbb{Z}$ is the agent's option inventory, $\tau_t \in \mathbb{R}_{\geq 0}$ is time to expiration, $q^h_t \in \mathbb{R}$ is the spot hedge position, $\text{cash}_t$ is accumulated cash, and $K_t$ is the current option strike (reset at each rollover). The agent directly observes all components of $s_t$.

Given oracle access to $\text{regime}_t$ and $\mathbf{P}$, the agent computes the belief-weighted BSM price as its fair value estimate:
$$
\hat{V}_t = \sum_i w_i \cdot \text{BSM}(S_t, K_t, \tau_t, r_f, \sigma_i)
$$
where $w_i = \mathbf{P}[\text{regime}_t, i]$ are the one-step-ahead transition probabilities. This is the forward-looking vol-of-vol blended price that the simulator uses internally, so the oracle agent's fair value estimate is exact up to Jensen's inequality in the BSM formula.

**Action Space $\mathcal{A}$**

The agent makes a joint continuous action at each timestep:
$$
a_t = (\delta_t,\ \Delta^{\text{target}}_{P,t})
$$

The half-spread $\delta_t \in [c \cdot S_t \cdot |\hat{\Delta}_t|,\ \hat{V}_t]$ determines how far each side of the quote is placed from $\hat{V}_t$. The lower bound ensures the spread covers at minimum the cost of delta-hedging a fill; the upper bound prevents a non-positive bid.

The hedge target $\Delta^{\text{target}}_{P,t} \in [\min(0, \hat{\Delta}_{P,t}),\ \max(0, \hat{\Delta}_{P,t})]$ specifies the desired residual portfolio delta after hedging. This constrains the hedge to reduce $|\hat{\Delta}_{P,t}|$ without flipping sign and introducing new directional exposure. The resulting hedge trade is $u^h_t = \hat{\Delta}_{P,t} - \Delta^{\text{target}}_{P,t}$ units of the spot.

**Transition Function $T(s' \mid s, a)$**

*Spot price:* Regime-switching geometric Brownian motion:
$$
S_{t+1} = S_t \cdot \exp\!\left(\left(\mu - \tfrac{1}{2}\sigma^{*2}_t\right)dt + \sigma^*_t\sqrt{dt}\,\varepsilon_t\right), \quad \varepsilon_t \sim \mathcal{N}(0,1)
$$

*Regime:* Markov chain transition via the Hardy (2001) matrix:
$$
\mathbf{P} = \begin{pmatrix} 0.9982 & 0.0018 \\ 0.0022 & 0.9978 \end{pmatrix}
$$
with $\sigma_L = 12.1\%$ and $\sigma_H = 26.9\%$ annualized.

*Time to expiry:* Decrements deterministically: $\tau_{t+1} = \max(\tau_t - dt, 0)$. At expiry, remaining inventory is settled at the terminal payoff $\max(S_T - K, 0)$, $q_t$ resets to zero, and a new at-the-money contract with $K = S_t$ and fresh $\tau_0$ begins.

*Inventory:* Updated by fills:
$$
q_{t+1} = q_t + f^b_t - f^a_t
$$
where fills follow an Avellaneda-Stoikov exponential intensity model: $f^b_t, f^a_t \sim \text{Bernoulli}(\lambda \, dt)$ with $\lambda = A e^{-k\delta_t}$.

**Reward Function $R(s, a)$**

The reward at each timestep is mark-to-market P&L net of a quadratic penalty on residual delta:
$$
r_t = d\text{PnL}_t - \varphi \cdot (\Delta^{\text{target}}_{P,t})^2
$$
$$
d\text{PnL}_t =
\underbrace{q_t \cdot dV^*_t}_{\text{mark-to-market}}
+ \underbrace{\delta_t \cdot (f^b_t + f^a_t)}_{\text{spread capture}}
+ \underbrace{u^h_t \cdot dS_t}_{\text{hedge P\&L}}
- \underbrace{c \cdot S_t \cdot |u^h_t|}_{\text{hedge cost}}
$$
where $dV^*_t = V^*_{t+1} - V^*_t$ is the change in true option fair value. The discount factor is $\gamma = 1$ since the finite episode horizon induced by $\tau_t \to 0$ renders discounting unnecessary.

**Parameter Table**

| Symbol | Meaning | Value |
|---|---|---|
| $dt$ | Timestep | $1/252$ (1 trading day) |
| $\mu$ | Spot drift | 0.05 (annualized) |
| $\sigma_L, \sigma_H$ | Regime volatilities | 0.121, 0.269 (annualized) |
| $\mathbf{P}$ | Hardy transition matrix | See above |
| $S_0$ | Initial spot price | 100 |
| $\tau_0$ | Initial time to expiry | 30 trading days |
| $A$ | Fill intensity scale | 140 |
| $k$ | Fill intensity decay | 6 |
| $c$ | Proportional hedge cost | 0.001 (10 bps) |
| $\varphi$ | Risk aversion | 0.01 |
| $\gamma$ | Discount factor | 1.0 |
| $N_{\text{contracts}}$ | Contracts per episode | 5 |
| $r_f$ | Risk-free rate | 0.05 |

---

## 4. Solution Approach

### 4.1 Analytical Benchmarks

Four analytical policies are evaluated as benchmarks, formed by combining two spread formulas with two hedging rules.

**Spread policies.** The *GLFT spread* adapts the half-spread to current portfolio risk:
$$
\delta^{\text{GLFT}}_t = \frac{\gamma}{2} \sigma^2_{\text{blend}} \tau_t \cdot |\hat{\Gamma}_{P,t}| \cdot S^2_t + \frac{1}{\gamma} \ln\!\left(1 + \frac{\gamma}{k}\right)
$$
where $\sigma^2_{\text{blend}} = \sum_i w_i \sigma_i^2$ uses the transition-weighted regime variance and $\hat{\Gamma}_{P,t}$ is the portfolio gamma. The second term is the symmetric Avellaneda-Stoikov component. This spread is clamped to the economically valid range $[c \cdot S_t \cdot |\hat{\Delta}_t|,\ \hat{V}_t]$. The *Naive spread* is a fixed half-spread of \$0.10.

**Hedging policies.** The *Whalley-Wilmott (WW) hedger* computes a no-trade halfwidth:
$$
H_t = \left(\frac{3c \cdot S^2_t \cdot |\hat{\Gamma}_{P,t}|}{2\varphi \sigma^2_{\text{blend}}}\right)^{1/3}
$$
and hedges only when $|\hat{\Delta}_{P,t}| > H_t$, targeting the band edge. The *Naive hedger* always targets $\Delta^{\text{target}} = 0$.

Both analytical benchmarks use the same oracle belief-weighted BSM fair value as the MCTS agent, so information is held equal across all policies.

### 4.2 MCTS-DPW Oracle MDP

The MCTS solver uses Double Progressive Widening (DPW) to handle the continuous action space. At each decision step, MCTS-DPW is called from a fresh root representing the current MDP state; the tree is not carried forward between timesteps.

**Action widening.** The number of unique actions tried at any node grows as $k \cdot n^\alpha$ where $n$ is the visit count. With $k = 2$ and $\alpha = 0.5$, this yields roughly $2\sqrt{N}$ actions after $N$ iterations, balancing exploration of the continuous action space against repeated rollouts from known actions.

**Action sampling.** New actions are sampled from the GLFT+WW closed-form solution with additive Gaussian noise. This informed prior concentrates sampling near analytically good actions while still exploring the neighborhood, rather than sampling uniformly over the full range.

**Rollout policy.** From the leaf node, a GLFT+WW rollout policy is used to estimate the value of the remaining episode. This substitutes domain knowledge for random rollouts and substantially reduces variance in value estimates.

**Solver parameters.** Each decision step runs 200 iterations with a planning depth of 20, UCB exploration constant of 1.0, and a rollout that follows GLFT+WW for the remaining steps of the episode.

The pseudocode for a single decision step is:

```
function mcts_action(s_t, mdp, config, vm):
    solver ← DPWSolver(n_iter=200, depth=20, k=2.0, α=0.5,
                       sampler=GaussianGLFT(config, vm),
                       rollout=GLFTWWRollout(config, vm))
    planner ← solve(solver, mdp)
    return action(planner, s_t)
```

---

## 5. Results

All results are from Monte Carlo evaluation over 1,000 episodes with seed 42. Each episode spans 5 consecutive at-the-money call options, each with a 30-trading-day lifetime (150 timesteps per episode). All agents, both analytical benchmarks and MCTS, receive oracle information: they observe the true current regime index and have access to the transition matrix $\mathbf{P}$.

Performance is reported as mean episode P&L, its standard deviation, and the Sharpe ratio $\mu/\sigma$ (no risk-free subtraction since the reward is already a net P&L). Supplementary statistics of mean half-spread $\bar{\delta}$, hedge trade frequency, mean absolute portfolio delta $|\hat{\Delta}_P|$, and total hedge transaction cost per episode separate the quoting and hedging contributions to performance.

### 5.1 Analytical Benchmarks

**Table 1.** Episode P&L summary  1,000 episodes, oracle information.

| Policy | Mean P&L | Std P&L | Sharpe | Mean $\delta$ | Hedge% | $\overline{|\hat{\Delta}_P|}$ | Hedge cost |
|---|---|---|---|---|---|---|---|
| *Constant volatility ($\sigma = 0.20$)* | | | | | | | |
| GLF-T + WW | \$3.971 | \$2.244 | 1.770 | \$0.387 | 36.9% | 0.089 | \$1.441 |
| GLF-T + Naive | \$3.640 | \$1.930 | **1.886** | \$0.383 | 69.4% | 0.071 | \$1.842 |
| Naive + WW | \$4.625 | \$4.054 | 1.141 | \$0.101 | 48.3% | 0.174 | \$4.598 |
| Naive + Naive | \$4.165 | \$3.430 | 1.215 | \$0.101 | 80.0% | 0.146 | \$5.178 |
| *Regime-switching (Hardy 2001)* | | | | | | | |
| GLF-T + WW | \$4.104 | \$2.319 | 1.770 | \$0.377 | 36.7% | 0.090 | \$1.507 |
| GLF-T + Naive | \$3.684 | \$2.001 | **1.841** | \$0.377 | 69.9% | 0.072 | \$1.884 |
| Naive + WW | \$4.306 | \$4.388 | 0.981 | \$0.101 | 48.2% | 0.171 | \$4.676 |
| Naive + Naive | \$3.869 | \$3.799 | 1.018 | \$0.101 | 79.9% | 0.146 | \$5.232 |

**Spread policy.** The GLFT spread, which scales with portfolio gamma, time to expiry, and variance, produces spreads averaging \$0.38  roughly 3.8× wider than the \$0.10 fixed spread used by the Naive policies. The much narrower fixed spread generates more fills and higher gross P&L on average, but at the cost of greater unhedged delta exposure, which manifests as larger $|\hat{\Delta}_P|$ and dramatically higher P&L variance (Figure 1 and Figure 2).

**Hedging policy.** The WW no-trade band allows $\hat{\Delta}_P$ to drift within a tolerance before hedging, resulting in a 37% hedge frequency and lower transaction costs. The Naive hedge always targets $\Delta^{\text{target}} = 0$, trading approximately twice as often and incurring higher costs, but keeping $|\hat{\Delta}_P|$ tighter (Figure 3).

**The GLF-T + Naive paradox.** The highest Sharpe ratio across both environments is achieved by GLF-T + Naive (1.886 constant vol, 1.841 Hardy), which combines the analytically derived spread with the simplest possible hedging rule. GLF-T + WW, despite incorporating a theoretically superior hedging policy, has lower Sharpe in both environments. The explanation lies in variance: aggressive full-hedging keeps the P&L distribution tighter (std \$1.93 vs. \$2.24 on constant vol), and Sharpe rewards this even though mean P&L is lower. In this parameterization the WW band width is wide enough to allow meaningful delta accumulation between hedges, which adds variance without sufficient compensating spread income. This is parameterization-specific  with higher $\varphi$ or wider spreads the WW band advantage would likely reassert itself.

**Regime-switching vs. constant vol.** Mean P&L is similar across both environments for the GLFT policies (\$3.97 vs. \$4.10, \$3.64 vs. \$3.68), indicating that the oracle vol calibration correctly accounts for the regime distribution. The Naive spread policies show a more pronounced decline on Hardy (\$4.62 → \$4.31, \$4.17 → \$3.87) because their fixed spread is not adjusted to compensate for additional volatility risk in the high-vol regime (Figures 4a, 4b).

### 5.2 MCTS-DPW Oracle MDP

**Table 2.** MCTS-DPW vs. best analytical benchmark  Hardy regime-switching, 1,000 episodes.

| Policy | Mean P&L | Std P&L | Sharpe |
|---|---|---|---|
| GLF-T + WW | \$4.104 | \$2.319 | 1.770 |
| GLF-T + Naive (best benchmark) | \$3.684 | \$2.001 | **1.841** |
| Naive + WW | \$4.306 | \$4.388 | 0.981 |
| Naive + Naive | \$3.869 | \$3.799 | 1.018 |
| **MCTS-DPW (oracle MDP)** | **\$2.697** | **\$2.805** | **0.962** |

Despite receiving identical oracle information as the analytical benchmarks and solving a fully observable MDP, MCTS-DPW underperforms all four benchmarks on both mean P&L and Sharpe.

**Why it underperforms.** The fundamental constraint is the computation budget. Each step, MCTS-DPW runs 200 iterations with a planning depth of 20 before handing off to a GLFT+WW rollout. With DPW parameters $k=2$, $\alpha=0.5$, 200 iterations produce approximately $k \cdot 200^\alpha \approx 28$ unique actions at the root, each sampled roughly 7 times. A rollout that runs for the remaining 130 steps of the episode introduces high variance in the value estimate. Seven samples drawn from a distribution with episode-level std of approximately \$3 cannot reliably rank actions whose expected value difference is on the order of cents per step.

In short, 200 MCTS queries per step is insufficient to empirically rediscover what GLFT derives analytically in closed form. The information is available  the state is fully observed and the transition model is correct  but the statistical sample size is too small to consistently select better-than-random actions at the root.

**What this finding implies.** This result does not indict MCTS as a method; it quantifies the computational cost of discovering structure that the analytical benchmarks encode for free. In this well-characterized domain, the analytical solutions distill decades of microstructure theory into a handful of formulas that are effectively impossible to match empirically at realistic compute budgets. An MCTS agent with 10× the query budget might approach, but is unlikely to surpass, the Sharpe of GLF-T + Naive without incorporating additional domain structure into the action prior.

---

## 6. Conclusion

This paper formulated options market making as a fully observable MDP and implemented a complete evaluation: four analytical benchmark policies and an MCTS-DPW oracle MDP solver, all evaluated on a Hardy (2001) regime-switching simulator with parameters calibrated to historical S&P 500 returns.

The central empirical finding is that **analytical structure dominates general-purpose search at realistic compute budgets**. The best analytical policy (GLF-T + Naive) achieves a Sharpe of 1.84 on the Hardy environment, while the MCTS oracle MDP  despite having identical information  achieves 0.96. The gap is not an information disadvantage; it is the cost of having to rediscover empirically what the GLFT derivation provides analytically. This has a direct practical implication: in well-studied domains where closed-form solutions exist for sub-problems, hybrid architectures  domain theory for the base policy, learned adjustments for the residual  will outperform pure search solvers at any realistic budget.

A secondary finding is that **the theoretically superior hedging policy (WW no-trade band) does not dominate naive full-hedging in this parameterization**. GLF-T + Naive beats GLF-T + WW on Sharpe in both environments because more frequent hedging reduces P&L variance enough to offset higher transaction costs. This illustrates a general point: theoretical optimality of a component policy does not guarantee optimality of the composite system when the performance metric is Sharpe rather than expected P&L.

---

## 7. Future Work

**Partial observability (POMDP extension).** The oracle MDP evaluated here provides a clean upper bound on what any fully informed agent can achieve. The natural next step is to remove oracle access and require the agent to infer $\sigma^*_t$ from the history of log returns alone, turning the problem into a POMDP. A particle filter over continuous volatility  treating $\sigma^*_t$ as an unknown scalar rather than assuming knowledge of the Hardy structure  provides the belief representation, and POMCPOW or QMDP can serve as the planner. The performance gap between the POMDP agent and the oracle MDP benchmarks then cleanly quantifies the cost of partial observability alone, holding policy approximation fixed.

**The two-volatility problem.** A more sophisticated agent should track two structurally different volatility objects separately. The simulator prices options using a transition-probability-weighted blend $\bar{\sigma}_t$ that reflects the risk-neutral measure's expectation over the regime distribution. The particle filter, by contrast, estimates instantaneous realized volatility from the return path  a physical-measure quantity. In regime-switching environments these diverge systematically when the market prices in the probability of an upcoming regime transition that has not yet materialized in returns. The correct architecture is to use $\hat{\sigma}^{\text{inst}}_t$ for Greek calculations (instantaneous sensitivity to current dynamics) and a flow-adjusted implied volatility $\hat{\sigma}^{\text{IV}}_t$ for quote centering (forward-looking fair value).

**Flow-adjusted fair value.** Following Glosten-Milgrom, fill asymmetry is a direct signal about $\hat{V}_t$ bias. A natural extension maintains a flow-adjusted fair value updated by fill imbalance:
$$
\hat{V}^{\text{flow}}_{t+1} = \hat{V}^{\text{flow}}_t + \eta \cdot (f^a_t - f^b_t) \cdot \delta_t
$$
where $\eta$ is a learning rate and the fill imbalance shifts the estimate upward on excess ask fills and downward on excess bid fills. At steady state, $\hat{V}^{\text{flow}}_t$ converges to the price at which the market is indifferent to buying or selling  the operational definition of fair value that requires no knowledge of the true pricing function.

**Larger MCTS budget and improved action priors.** The current 200-iteration budget is too small to reliably rank continuous actions at this problem scale. Scaling the query budget by 10× or incorporating tighter domain-informed action distributions (e.g., sampling $\delta$ from the GLFT formula with regime-specific variance rather than a flat Gaussian perturbation) would likely close a meaningful fraction of the gap to the analytical benchmarks.

---

## 8. Contributions and Release

This paper and all of its contributions are from Kyan Nelson exclusively.

**Algorithm implementations.** The following were implemented from scratch: the Hardy regime-switching GBM simulator, the BSM pricing and Greeks module, the Avellaneda-Stoikov fill model, the GLFT spread formula, the Whalley-Wilmott no-trade band, the portfolio accounting and reward function, the MCTS-DPW oracle MDP solver (via the POMDPs.jl and MCTS.jl Julia packages for solver scaffolding, with custom action samplers and rollout policy), and all evaluation and figure generation code. The MCTS.jl and POMDPs.jl libraries provided the DPW solver infrastructure; the core domain logic (MDP definition, action sampling, rollout) was written by the author.

The authors grant permission for this report to be posted publicly.

---

## References

[1] F. Black, M. Scholes, "The Pricing of Options and Corporate Liabilities," *Journal of Political Economy*, 81(3), 637–654, 1973.

[2a] M. Avellaneda, S. Stoikov, "High-frequency trading in a limit order book," *Quantitative Finance*, 8(3), 217–224, 2008.

[2b] O. Guéant, C.-A. Lehalle, J. Fernandez-Tapia, "Dealing with the Inventory Risk: A Solution to the Market Making Problem," *Mathematics and Financial Economics*, 7(4), 477–507, 2013.

[3] L. Glosten, P. Milgrom, "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders," *Journal of Financial Economics*, 14(1), 71–100, 1985.

[4] A. Whalley, P. Wilmott, "An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs," *Mathematical Finance*, 7(3), 307–324, 1997.

[5] M. Hardy, "A Regime-Switching Model of Long-Term Stock Returns," *North American Actuarial Journal*, 5(2), 41–53, 2001.

---

## Appendix: Figures

**Figure 1**  P&L distributions across all four analytical benchmark policies under constant volatility (left) and Hardy regime-switching (right). GLFT spread policies produce tighter, higher-Sharpe distributions; Naive spread policies have wider tails and higher mean P&L at the cost of variance.

![fig1_pnl_distributions](/Users/n.kyan/Spring 2026/Academic/DMU/Final_Project/results/fig1_pnl_distributions.png)

**Figure 2**  Mean half-spread as a function of days to expiry for each policy. GLFT spreads follow a hump-shaped curve reflecting the gamma term structure: gamma is highest at intermediate $\tau$ for at-the-money options. Naive spreads are flat at \$0.10.

![fig2_spread_vs_tau](/Users/n.kyan/Spring 2026/Academic/DMU/Final_Project/results/fig2_spread_vs_tau.png)

**Figure 3**  Hedge frequency (left) and mean absolute net portfolio delta (right) for each policy across both environments. GLFT+WW trades least frequently and accumulates more delta on average; GLFT+Naive trades more but keeps delta tighter.

![fig3_hedge_behavior](/Users/n.kyan/Spring 2026/Academic/DMU/Final_Project/results/fig3_hedge_behavior.png)

**Figure 4a**  Cumulative P&L trajectories for 5 representative episodes under constant volatility. GLFT spread policies accumulate positive P&L steadily; Naive spread policies show high variance including large drawdown episodes.

![fig4a_cumulative_pnl_const](/Users/n.kyan/Spring 2026/Academic/DMU/Final_Project/results/fig4a_cumulative_pnl_const.png)

**Figure 4b**  Cumulative P&L trajectories for 5 representative episodes under Hardy regime-switching. Shaded regions indicate high-volatility regime periods. The Naive+WW policy suffers particularly large drawdowns during sustained high-vol episodes, while GLFT spread policies remain broadly profitable across both regimes.

![fig4b_cumulative_pnl_hardy](/Users/n.kyan/Spring 2026/Academic/DMU/Final_Project/results/fig4b_cumulative_pnl_hardy.png)

**Figure 5** Single-episode trajectory comparison between MCTS-DPW and GLF-T+WW on a shared Hardy environment seed. (a) Spot price with high-vol regime shaded. (b) Half-spread $\delta $: GLFT quotes a smooth, stable spread around \$0.40–\$0.60 while MCTS exhibits erratic spikes above \$4, a symptom of noisy Q-value estimates at 200 iterations. (c) Hedge target $\Delta_{\text{target}} $: both policies largely remain near zero with MCTS showing larger excursions. (d) Cumulative P&L: both policies finish positive on this episode with MCTS tracking near GLFT despite noisier actions.

![fig5_mcts_trajectory](/Users/n.kyan/Spring 2026/Academic/DMU/Final_Project/results/fig5_mcts_trajectory.png)

**Figure 6** P&L distribution comparison between MCTS-DPW and GLF-T+WW over 1,000 Hardy episodes. MCTS mean ($2.73, Sharpe 0.98) is shifted left relative to GLF-T+WW ($4.10, Sharpe 1.77), with a heavier left tail. The distributions overlap substantially, confirming that MCTS does not fail catastrophically but consistently underperforms due to policy noise rather than systematic directional error.

![fig6_mcts_pnl_comparison](/Users/n.kyan/Spring 2026/Academic/DMU/Final_Project/results/fig6_mcts_pnl_comparison.png)

**Figure 7** Action scatter plots comparing MCTS-DPW and GLF-T+WW across a full episode. (a) Spread vs. time-to-expiry: GLFT traces the expected hump-shaped gamma term structure curve while MCTS spreads are scattered broadly at all $\tau $ with no coherent structure. (b) Spread vs. $|\Delta_{\text{target}}| $: GLFT spread grows with inventory exposure as the formula prescribes; MCTS shows no such relationship. Both panels confirm that 200 iterations are insufficient to recover the inventory-risk scaling structure encoded analytically in GLFT.

![fig7_mcts_action_scatter](/Users/n.kyan/Spring 2026/Academic/DMU/Final_Project/results/fig7_mcts_action_scatter.png)
