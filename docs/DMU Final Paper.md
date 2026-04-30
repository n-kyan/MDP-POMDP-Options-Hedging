## 1. Introduction

the agent observes everything a real market maker could see, and nothing it could not.

So the real question is not "should I use any models" but **"which models should inform which parts of the system, and where should the agent be free to discover its own behavior."**

I have concluded that pure model-free reinforcemnt learning (RL) is not achievable. Black-Scholes-Merton (BSM) is absolutely required here for the calculation of inventory risk greeks. These greeks are what the agent will observe to inform both its hedging and quoting actions. Given that I must use BSM, the door is now open to incorporate other analytical model in the problem formulation. 

## 2. Background and Related Work

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

## 8. Contributions and Release

This paper and all of its contributions are from Kyan Nelson exclusively.

The authors grant permission for this report to be posted publicly.