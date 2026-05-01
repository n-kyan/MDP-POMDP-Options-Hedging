## 1. Introduction

the agent observes everything a real market maker could see, and nothing it could not.

So the real question is not "should I use any models" but **"which models should inform which parts of the system, and where should the agent be free to discover its own behavior."**

I have concluded that pure model-free reinforcemnt learning (RL) is not achievable. Black-Scholes-Merton (BSM) is absolutely required here for the calculation of inventory risk greeks. These greeks are what the agent will observe to inform both its hedging and quoting actions. Given that I must use BSM, the door is now open to incorporate other analytical model in the problem formulation. 

Options market making (OMM) is a sequential decision problem under partial observability. A market maker (MM) continuously posts a bid and an ask around an estimated fair value $\hat{V}_t $, earning the spread when trades execute against their quotes. The core tension is immediate: tighter spreads attract more order flow and increase fill frequency, but they reduce per-trade revenue and expose the MM to inventory accumulation. Wider spreads generate more revenue per fill but reduce flow and risk being consistently undercut by competitors.

What makes OMM genuinely difficult — and genuinely a POMDP problem — is not the spread-width tradeoff itself but the structure of the uncertainty the MM operates under. The two quantities most critical to the quoting and hedging decisions are the true fair value of the option $V^*_t $ and the true instantaneous volatility $\sigma^*_t $. Neither is observable. $V^*_t $ must be inferred from noisy market prices; $\sigma^*_t $ must be estimated from the history of log returns. Every pricing and risk computation the MM performs is downstream of these latent quantities, which means every quote is placed under epistemic uncertainty about the very inputs that determine whether that quote is profitable.

This uncertainty is epistemic rather than aleatoric. It arises from incomplete information rather than intrinsic randomness, and it is reducible through observation. Each log return narrows the posterior over $\sigma^*_t $; each fill event updates the agent's sense of where the market believes fair value is. This is precisely the setting POMDPs are designed to handle: the agent cannot observe the true state directly, but it can maintain a belief over the state and update it with each observation, conditioning its policy on that belief.

The analytical literature on market making has developed strong models for each sub-problem in isolation: Black-Scholes-Merton (BSM) for option valuation, Avellaneda-Stoikov (AS) and Guéant-Lehalle-Fernandez-Tapia (GLFT) for optimal spread-setting under inventory risk, Glosten-Milgrom for adverse selection pricing, and Whalley-Wilmott for hedging under transaction costs. A companion paper [Nelson, 2025] showed through simulation that these benchmarks perform well when volatility is known but diverge meaningfully under regime-switching, and that no single benchmark integrates all four decisions. The binding constraint is the absence of a mechanism for jointly inferring the latent state and acting on that inference.

This paper implements a POMDP solution to the OMM problem. The agent maintains a continuous particle filter belief over $\sigma^*_t $, uses that belief for BSM-based Greek calculations and fair value estimation, and selects spread and hedge actions via Monte Carlo Tree Search with Double Progressive Widening (MCTS-DPW). The agent is model-agnostic with respect to the volatility process — it does not know the regime-switching structure that generates the simulation data — and must discover its volatility belief purely from observed log returns. The POMDP policy is then benchmarked against the analytical solutions to quantify the value of closed-loop belief-based decision-making.

## 2. Background and Related Work

**Options pricing and the BSM framework.** Black, Scholes, and Merton (1973) derived the canonical closed-form pricing formula for European options under the assumptions of continuous trading, log-normally distributed prices, and constant volatility. Despite these assumptions being empirically violated, BSM remains the industry-standard tool for computing option fair values and Greeks — the first-order sensitivities ($\Delta $, $\Gamma $) that characterize how an option position changes with the underlying. In this work, BSM is not treated as the pricing model for the simulator but as the agent's inference tool: the agent uses BSM under its believed volatility $\hat{\sigma}_t $ to compute portfolio Greeks that serve as observations and to anchor its fair value estimate $\hat{V}_t $.

**Optimal quoting under inventory risk.** Avellaneda and Stoikov (2008) formulated the market maker's spread-setting problem as a stochastic control problem, deriving a reservation price that adjusts for inventory and an optimal spread as a function of inventory, time horizon, and risk aversion. Guéant, Lehalle, and Fernandez-Tapia (2013) extended this to a tractable closed-form solution (GLFT) with exponential fill intensity, giving the MM explicit formulas for bid and ask offsets. GLFT is used as the quoting benchmark in this paper. Its key limitation in the OMM context is that it assumes known volatility; under regime-switching the inventory-risk penalty it computes is miscalibrated whenever the MM's volatility belief is wrong.

**Adverse selection and informed order flow.** Glosten and Milgrom (1985) modeled the bid-ask spread as an equilibrium outcome of a game between market makers and traders who are either informed (with private knowledge of true value) or uninformed (liquidity-motivated). A central insight is that fill events are Bayesian signals: an ask fill is evidence that the buyer believes the true value exceeds the ask, and the MM should update their fair value estimate upward accordingly. While the current implementation delegates adverse selection protection to the spread width rather than maintaining a separate flow-adjusted fair value, the Glosten-Milgrom framework motivates the observation model structure and points toward a natural future extension (see Section 7).

**Hedging under transaction costs.** Whalley and Wilmott (1997) solved the problem of delta hedging with proportional transaction costs, showing that the optimal strategy is not continuous rebalancing but a no-trade band around the target delta. Hedging only when the portfolio delta drifts outside the band minimizes expected costs while bounding delta exposure. This is used as the hedging benchmark and directly informs the hedge action space: the agent's hedge target is constrained to reduce $|\hat{\Delta}_{P,t}| $ without flipping sign, which operationalizes the spirit of the WW band in a learned policy.

**Regime-switching volatility.** Hardy (2001) demonstrated that a two-state regime-switching model fits historical S&P 500 returns better than GARCH by the Schwarz-Bayes Criterion, estimating regime volatilities of approximately 12.1% and 26.9% annualized with daily transition probabilities that produce regimes lasting on the order of months. The simulation environment in this paper uses Hardy's estimated parameters directly. Critically, the agent does not observe the regime label or know the transition matrix — it models volatility as a continuous scalar with a log-normal random walk prior, and the particle filter must recover a useful volatility estimate from the return path alone.

**OMM as a POMDP.** The POMDP framework (Kaelbling, Littman, and Cassandra, 1998) formalizes decision-making under partial observability: the agent maintains a belief $b_t(s) = P(s_t = s \mid h_t) $ over the hidden state given its action-observation history, and the optimal policy maps beliefs to actions. Prior work on RL for market making has largely focused on the fully observable spot market-making case [Spooner et al., 2018] or used deep RL with raw market features as a proxy for belief [Ganesh et al., 2019]. The contribution here is an explicit POMDP formulation of the OMM problem with a principled belief representation — a particle filter over continuous volatility — coupled to an online tree search solver, bridging the microstructure literature and the POMDP algorithmic toolkit developed in this course.

## 3. Problem Formulation

**State Space $\mathcal{S}$**

The true state of the world is defined as:
$$
s_t=(S_t, V^*_t, \sigma^*_t, q_t, \tau_t)
$$
Where $S_t \in \mathbb{R}_{>0}$ is the underlying spot price, $V^*_t$ is the true fair value of the option, $\sigma^*_t$ is the true instantaneous volatility, $q_t \in \mathbb{Z}$ is the agent's option inventory,  and $\tau_t \in \mathbb{R}_{\geq 0}$ is time to expiration. Of these, both $\sigma^*_t$ and $V^*_t$ are  unobservable and crucial to the quoting action. Volatility is also a key input into the Black-Scholes formula which will be used to inform the agent of its current inventory risk. $V^*_t$ is included as an independent state component rather than a derived quantity because the agent does not assume knowledge of the pricing function used by the simulator. From the agent's perspective, $V^*_t$ is a latent quantity inferred jointly with $\sigma^*_t$.

**Observation Space $\mathcal{O}$**

At each timestep the agent will observe:
$$
o_t = (r_t, \hat \Delta_{P, t}, \Gamma_{P, t}, f_t, \tau_t)
$$
Where $r_t = \log(S_t / S_{t-1})$ is the log return of the spot and serves as the primary signal for the volatility belief. $\hat  \Delta_{P,t} = q_t \cdot \hat \Delta_t$ and  $\hat \Gamma_{P,t} = q_t \cdot \hat \Gamma_t$ are portfolio-level Greeks computed under the agent's  current believed volatility $\hat{\sigma}_t$. $\Delta$ is the rate at which $V_t$ changes when $S_t$ changes; $\Gamma$ is the rate at which $\Delta$ changes when $S_t$ changes. $\tau_t$ is time to expiration and $f_t \in \{-1, 0, +1\}$ is the fill indicator encoding ask fill, no fill, and bid fill respectively.

**Action Space $\mathcal{A}$**

The agent must make a joint action of:
$$
a_t = (\delta_t, \Delta^{\text{target}}_{P,t})
\\\\
\delta_t \in [c\cdot S_t \cdot |\hat \Delta_t|, \hat V_t]
\\\\
\Delta^{\text{target}}_{P,t} \in [\text{min}(0,\hat \Delta_{P,t}), \text{max}(0, \hat \Delta_{P, t})]
$$
The half-spread $\delta_t \in [c\cdot S_t \cdot |\hat \Delta_t|, \hat V_t]$ determines how far each side of the quote is placed from $\hat V_t$. These bounds are economic in natura as the half-spread must at least cover the cost of delta-hedging the position that would result from a fill and cannot be higher than the value of the option itself or the bid would be non-positive.

The hedge target $\Delta^{\text{target}}_{P,t} \in [\text{min}(0,\hat \Delta_{P,t}), \text{max}(0, \hat \Delta_{P, t})]$ specifies the desired  residual portfolio delta after hedging. This constrains the hedge to reduce $|\hat\Delta_{P,t}|$ without flipping its sign and introducing new directional exposure. The actual resulting trade is $u^h_t = \hat \Delta_{P, t}-\Delta^{\text{target}}_{P, t}$, where $u^h_t$ is the units of the spot to trade.

**Transition Function $T(s'|s, a)$**

$S_t$: Regime-switching GBM:
$$
S_{t+1}=S_t\cdot e^{((\mu-\frac{1}{2}\sigma^{*2})dt+\sigma^*_t \sqrt{dt}\varepsilon_t)}, \varepsilon_t \sim \mathcal{N}(0, 1)
$$
$\sigma^*_t$: Markov chain transition via Hardy (2001) matrix:
$$
\mathbf{P} = \begin{pmatrix} 0.9982 & 0.0018 \\ 0.0022 & 0.9978 \end{pmatrix}
$$
With $\sigma_L = 12.1\%$ and $\sigma_H = 26.9\%$ annualized. The agent does not observe  $\sigma^*_t$ or have knowledge of $\mathbf{P}$.

$\tau_t$: Decrements deterministically to zero:
$$
\tau_{t+1} = \text{max}(\tau_t-dt, 0)
$$
At expiry, remaining inventory is settled at the terminal payoff $\max(S_T - K, 0)$, $q_t$ resets to zero, and a new at-the-money contract with $K=S_t$ and fresh $\tau$ begins. Each episode spans multiple option lifetimes to expose the agent to the full term structure of risk.

$q_t$: Increments on bid fills and decrements on ask fills:
$$
q_{t+1} = q_t + f_t^b - f_t^a
$$
Where $f_t^b$ and $f_t^a$ are independent Bernoulli random variables:
$$
f_t^b \sim \text{Bernoulli}(\lambda dt)\\\\\
f_t^a \sim \text{Bernoulli}(\lambda dt)
$$
Where fill intensity is defined as:
$$
\lambda = Ae^{-k\delta_t}
$$
Where $A$ represents the overall market activity level and $k$ represents the market's sensitivity to changes in the bid-ask spread $\delta_t$.



**Reward Function $R(s, a)$**

The agent's reward at each timestep is realized P\&L net of a quadratic penalty on residual delta exposure scaled by a risk aversion parameter:
$$
r_t=d\text{PnL}_t-\varphi⋅(\Delta^{\text{target}}_{P,t})^2
$$

$$
d\text{PnL}_t = 
\underbrace{q_t \cdot d V^*_t}_{\text{mark-to-market}} 
+ \underbrace{\delta_t \cdot (f^b_t + f^a_t)}_{\text{spread capture}} 
+ \underbrace{u^h_t \cdot dS_t}_{\text{hedge P\&L}} 
- \underbrace{c \cdot S_t \cdot |u^h_t|}_{\text{hedge cost}}
$$

Where $dV^*_t = V^*_{t+1} - V^*_t$ is the change in true option fair value,  $\varphi$ is the risk aversion parameter, $c$ is the proportional transaction cost,  and $\Delta^{\text{target}}_{P,t}$ is the agent's chosen residual exposure. The discount factor is set to $\gamma = 1$ since the finite episode horizon induced by $\tau_t \to 0$ renders discounting unnecessary, and total undiscounted P\&L  is the natural performance metric for a single trading session. The true value of the option is used to calculate PnL reflecting how the market value of the option portfolio truly changes while the believed portfolio delta is used as for the risk penalty so that the agent is penalized for how much risk it believes it is taking on.



**Observation Model $Z(o_t \mid s'_t, a_{t-1})$**

The observation returns the probability distribution over possible observations given that the world is in state $s_t$ and the agent's previous action $a_{t-1}$. The agent maintains a belief over the true volatility $\sigma_t^*$, which drives both its quoting decisions and its BSM-based computations.

The sole component of $o_t$ whose distribution depends on $\sigma_t^*$ is the log return $r_t$. All other components are either deterministic functions of the observable state ($\tau_t$, $\hat \Delta_{P,t}$, $\hat \Gamma_{P,t}$) or are driven by the agent's own action ($f_t$). The observation likelihood used for belief updating is therefore:
$$
Z(o_t \mid s_t, a_{t-1}) = p(r_t \mid \sigma_t^*) = \mathcal{N}\left(r_t;\ \left(\mu - \tfrac{1}{2}\sigma_t^{*2}\right)dt,\ \sigma_t^{*2}dt\right)
$$
The agent models log returns as normally distributed, via the assumption embedded in GBM. While empirical returns are known to violate normality, the simulation generates returns from a GBM process, so the agent benefits from a fully correct observation model for this environment. The remaining observations inform the policy directly but do not contribute to the belief update over $\sigma_t^*$.

**Other information **

The agent's pricing model is doubly-approximate relative to the true process, both because volatility switches and because $V^*$ uses a probability-weighted volatility rather than the realized regime volatility.

Quote skewing is omitted from the action space; inventory management is delegated to the hedge action. This reduces the action space dimensionality at the cost of foregoing the spread-skewing channel of inventory control.

Although the simulator generates volatility from a two-state regime-switching process, the agent treats $\sigma^*_t$ as an unknown continuous quantity. The belief $b_t$ is a distribution represented numerically by a particle filter.

**Parameter Table:**

| Symbol                 | Meaning                                | Value                                                        |
| ---------------------- | -------------------------------------- | ------------------------------------------------------------ |
| $dt$                   | Timestep                               | 1/252 (1 trading day)                                        |
| $\mu$                  | Spot drift                             | 0.05 (annualized) (equal to risk free rate)                  |
| $\sigma_L, \sigma_H$   | Regime volatilities                    | 0.121, 0.269 (annualized)                                    |
| $\mathbf{P}$           | Hardy regime transition matrix         | $\begin{pmatrix} 0.9982 & 0.0018 \\ 0.0022 & 0.9978 \end{pmatrix}$ |
| $S_0$                  | Initial spot price                     | 100                                                          |
| $K$                    | Strike (reset to ATM at each rollover) | $S_0$ at start; $S_t$ at rollover                            |
| $\tau_0$               | Initial time to expiration             | 30/365 yr                                                    |
| $A$                    | Fill intensity scale                   | 100 (num of expected fills per side per day if $\delta=0$)   |
| $k$                    | Fill intensity decay                   | 3                                                            |
| $c$                    | Proportional hedge transaction cost    | 0.001 (10 bps)                                               |
| $\varphi$              | Risk aversion                          | 0.01                                                         |
| $\gamma$               | Discount factor                        | 1.0                                                          |
| $N_{\text{contracts}}$ | Contracts per episode                  | 5                                                            |
| $r_f$                  | Risk-free rate                         | 0.05                                                         |



## 4. Solution Approach

It is crucial to reiterate that the agent and simulation environment have been modelled independently. In order to better approximate the problems that real market makers face the agent observes everything a real market maker could see, and nothing it could not.

**On the GBM likelihood and the cheating question**

The particle filter uses a GBM log-normal likelihood as its observation model, consistent with the simulation's data generating process. This is a modeling choice, not a revelation of ground truth — GBM is a standard and widely-used model for equity returns, and its use here does not constitute information leakage. In a real deployment the likelihood function would be replaced with whatever return model is appropriate for the instrument — stochastic vol, jump-diffusion, or an empirically calibrated distribution — with no changes to the particle filter architecture.

## 5. Results

## 6. Conclusion

## 7. Future Work

The current agent maintains a single volatility estimate $\hat{\sigma}_t $ via a particle filter over log returns, which is then used for both fair value computation and Greek calculations. This conflates two distinct quantities that a more sophisticated agent should track separately.

**The two-volatility problem.** The simulator prices options as $V^*_t = \text{BSM}(S_t, K, \tau_t, r_f, \bar{\sigma}_t) $ where $\bar{\sigma}_t $ is a transition-probability-weighted blend of both regime volatilities — a forward-looking quantity that incorporates the market's expectation of future vol dynamics. The agent's particle filter, by contrast, estimates current instantaneous realized volatility from the path of log returns. These two quantities are structurally different objects. $\bar{\sigma}_t $ reflects the risk-neutral measure's expectation over the full regime distribution; $\hat{\sigma}^{\text{inst}}_t $ reflects the physical measure's best estimate of the current regime. In a regime-switching world they will systematically diverge whenever the market is pricing in the possibility of a regime transition that has not yet materialized in returns.

**Flow-adjusted fair value.** The Glosten-Milgrom framework implies that fill asymmetry is a direct signal about $\hat{V}_t $ bias: if asks are consistently being lifted more than bids, the agent's quote center is below $V^*_t $. A natural extension is to maintain a separate flow-adjusted fair value $\hat{V}^{\text{flow}}_t $ that is initialized to $\text{BSM}(S_t, K, \tau_t, r_f, \hat{\sigma}^{\text{inst}}_t) $ but updated by fill imbalance:
$$
\hat{V}^{\text{flow}}_{t+1} = \hat{V}^{\text{flow}}_t + \eta \cdot (f^a_t - f^b_t) \cdot \delta_t
$$
where $\eta $ is a learning rate and the fill imbalance $(f^a_t - f^b_t) $ shifts the estimate upward on excess ask fills and downward on excess bid fills. At steady state, $\hat{V}^{\text{flow}}_t $ converges to the price at which the market is indifferent to buying or selling — the operational definition of fair value that requires no knowledge of the true pricing function. An implied volatility $\hat{\sigma}^{\text{IV}}_t = \text{BSM}^{-1}(\hat{V}^{\text{flow}}_t) $ can then be backed out from this flow-adjusted estimate.

**Correct usage of each estimate.** The two volatility objects serve different roles. $\hat{\sigma}^{\text{IV}}_t $ should govern quote centering — it is the best estimate of the price at which the market clears, incorporating the forward-looking regime premium that pure return observation misses. $\hat{\sigma}^{\text{inst}}_t $ should govern Greek calculations — $\Delta $, $\Gamma $, and the Whalley-Wilmott no-trade band all describe instantaneous sensitivities under current realized dynamics, not forward-looking risk-neutral expectations. Using a forward-looking implied vol for Greeks would systematically over-hedge in low-vol regimes and under-hedge in high-vol regimes, since the Greeks would reflect the blended regime expectation rather than the realized local dynamics.

The current implementation uses $\hat{\sigma}^{\text{inst}}_t $ for both purposes, which is correct for Greeks but introduces a systematic downward bias in $\hat{V}_t $ relative to $V^*_t $ during low-volatility regimes when the transition probability assigns non-negligible weight to the high-vol state. Quantifying this bias and its P&L impact is a direct extension of the evaluation framework developed here.

## 8. Contributions and Release

This paper and all of its contributions are from Kyan Nelson exclusively.

The authors grant permission for this report to be posted publicly.







I agree with using MCTS.jl with DPW  (Double Progressive Widening) for the MDP phase. You are wrong about the particle filter implementation. it is a particle filter over continuois vol level not over regime_idx - that is just how the vol is generated. The agent should have to come to a belief about what it thinks the level of current volatility is. This believed volatility is what will be used for the greek calculations.

**1. MDP phase solver:** MCTS+DPW.

**2. POMDP agent model structure:** Model-agnostic continuous particle filter. The agent does NOT know it's a 2-regime Hardy process. It models vol as a continuous scalar with log-normal random walk: $\log\sigma_{t+1} = \log\sigma_t + \eta_t $, $\eta_t \sim \mathcal{N}(0, \xi^2 dt) $. Particles represent a distribution over continuous $\sigma $, not discrete regimes. `regime_idx` is simulator-only and never touches the agent model.

**3. Discount factor:** $\gamma = 1.0 $.

**4. Compute budget:** Start with `n_queries = 50`, `max_depth = 5`. Leave both as tunable parameters.