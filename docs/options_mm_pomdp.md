# Options Market Making as a POMDP

## Problem Overview

A market maker continuously posts bid and ask quotes on options, earns the spread on fills, and manages the resulting Greek exposure through spot hedging. The core challenge is that the market maker cannot directly observe the true mid price, latent volatility, the current market regime, or the toxicity of incoming order flow — making this a fundamentally partially observable problem.

The classical Avellaneda-Stoikov framework addresses spot market making but collapses the hedging decision and ignores regime uncertainty and adverse selection. The POMDP formulation makes all of these explicit and treats quote placement and hedging as a joint decision under uncertainty.

---

## The POMDP Tuple $(S, A, O, T, Z, R, \gamma)$

### State Space $\mathcal{S}$

$$s = (S_t,\ m_t,\ \sigma_t,\ r_t,\ \Gamma_t,\ \lambda_t^{\text{inf}},\ \tau)$$

| Variable | Description | Observable? |
|---|---|---|
| $S_t$ | Spot price of the underlying | Yes (low noise) |
| $m_t$ | True mid price of the option | No — inferred from quotes |
| $\sigma_t$ | Latent realized volatility | No — inferred from returns and implied vol |
| $r_t$ | Discrete market regime (e.g. low vol / high vol / crisis) | No — inferred from all signals |
| $\Gamma_t$ | Current Greek exposure vector $(\Delta, \Gamma, \mathcal{V}, \Theta)$ | Yes — known from own book |
| $\lambda_t^{\text{inf}}$ | Informed order flow intensity (adverse selection risk) | No — inferred from fill patterns |
| $\tau$ | Time to expiry | Yes — deterministic |

**Key insight:** $S_t$, $\Gamma_t$, and $\tau$ are fully observable. The genuinely hidden components are $m_t$, $\sigma_t$, $r_t$, and $\lambda_t^{\text{inf}}$. This structure motivates a Rao-Blackwellized belief updater.

---

### Action Space $\mathcal{A}$

$$a = (\delta^b_t,\ \delta^a_t,\ h_t)$$

| Variable | Description |
|---|---|
| $\delta^b_t$ | Bid offset from mid (how aggressively to post the bid) |
| $\delta^a_t$ | Ask offset from mid (how aggressively to post the ask) |
| $h_t$ | Target delta after hedging: $h_t \in \{-0.25, -0.20, \ldots, 0.20, 0.25\}$ |

**Design notes:**
- Quote size $q$ is fixed (simplification — can be extended to a third quote dimension)
- $h_t$ is expressed as a target delta rather than a trade size, which is more natural from a risk management perspective
- Quote placement and hedging are joint actions at the same timestep. The agent implicitly learns to hedge infrequently because hedging incurs transaction costs in the reward function — the frequency difference emerges from the reward structure rather than being hardcoded

---

### Observation Space $\mathcal{O}$

$$o = (S_t,\ \hat{\sigma}_t,\ p_t^b,\ p_t^a,\ f_t,\ V_t,\ \Gamma_t,\ \tau)$$

| Variable | Description | What it reveals |
|---|---|---|
| $S_t$ | Spot price | Direct, low-noise signal of underlying |
| $\hat{\sigma}_t$ | Implied volatility from market option prices | Noisy signal of latent $\sigma_t$; gap between implied and realized is tradeable |
| $p_t^b, p_t^a$ | Market best bid and ask | Noisy signal of true mid $m_t$ via midpoint |
| $f_t$ | Fill events (hit bid / lifted ask / size) | Most informative signal for $\lambda_t^{\text{inf}}$ — adverse fills reveal informed flow |
| $V_t$ | Recent trade volume and order arrival intensity | Feeds Hawkes posterior on flow toxicity |
| $\Gamma_t$ | Own Greek exposure | Fully observable from own book |
| $\tau$ | Time to expiry | Deterministic |

**Note on fill events:** $f_t$ is the richest observation. The conditional distribution of price moves *given a fill* is the key signal — if you are filled and the price immediately moves against you, this is strong evidence that $\lambda_t^{\text{inf}}$ is elevated. This is the Lee-Ready logic embedded in the observation model.

---

### Transition Model $T(s' \mid s, a)$

Each state component evolves as follows:

**Spot price** — fat-tailed diffusion with latent vol:
$$S_{t+1} = S_t \exp\!\left(\mu \Delta t + \sigma_t \sqrt{\Delta t}\, \epsilon_t\right), \quad \epsilon_t \sim \text{Student-}t$$

**Latent volatility** — stochastic volatility model with regime-dependent parameters:
$$\log \sigma_{t+1} = \mu_{r_t} + \phi_{r_t}(\log \sigma_t - \mu_{r_t}) + \eta_t$$

Each regime $r_t$ has its own mean vol level $\mu_{r_t}$ and persistence $\phi_{r_t}$. No Gaussian assumption on $\eta_t$.

**Regime** — discrete Markov chain:
$$P(r_{t+1} = j \mid r_t = i) = \Pi_{ij}$$

where $\Pi$ is the regime transition matrix.

**Informed flow intensity** — Hawkes self-exciting process:
$$\lambda_{t+1}^{\text{inf}} = \mu_\lambda + (\lambda_t^{\text{inf}} - \mu_\lambda)e^{-\kappa \Delta t} + \sum_{\text{fills in }\Delta t} \alpha_\lambda$$

Baseline intensity $\mu_\lambda$, mean reversion at rate $\kappa$, each fill event excites the intensity by $\alpha_\lambda$.

**Greeks** — deterministic given state and positions. Hedge action directly modifies delta:
$$\Delta_{t+1} = \Delta_t - h_t$$

**Time to expiry** — deterministic countdown: $\tau \rightarrow \tau - \Delta t$

---

### Observation Model $Z(o \mid s)$

**Spot price:** nearly direct observation with tight noise.

**Implied vol:** noisy observation of latent vol with regime-dependent risk premium:
$$\hat{\sigma}_t = \sigma_t + \text{RiskPremium}_{r_t} + \epsilon_\sigma$$

The risk premium is regime-dependent — in crisis regimes implied vol systematically overshoots realized vol.

**Market bid/ask:** noisy signals of true mid, with spread endogenously depending on vol and informed flow intensity:
$$p_t^b = m_t - \tfrac{1}{2}\text{spread}_t + \epsilon_b, \quad p_t^a = m_t + \tfrac{1}{2}\text{spread}_t + \epsilon_a$$

**Fill probability:** probability of being filled at the posted bid depends on quote aggressiveness, informed flow, and overall order arrival intensity:
$$P(f_t^b = 1 \mid s) = g\!\left(m_t - p_t^b,\ \lambda_t^{\text{inf}},\ V_t\right)$$

---

### Reward Function $R(s, a)$

$$R(s, a) = R^{\text{spread}} + \theta_t \Delta t - C^{\text{hedge}} - C^{\text{inventory}} - C^{\text{adverse}}$$

**Spread income:**
$$R^{\text{spread}} = f_t^b \cdot \delta^b_t \cdot q + f_t^a \cdot \delta^a_t \cdot q$$

Earned only on fills. Tighter quotes fill more but earn less per fill.

**Theta decay income:**
$$+\ \theta_t \cdot \Delta t$$

Passive income from time decay. As a short options market maker you earn theta every timestep. This directly offsets inventory risk in quiet markets — capturing the core short gamma / long theta tradeoff.

**Hedging cost:**
$$C^{\text{hedge}} = |h_t - \Delta_t| \cdot c_{\text{trade}}$$

Transaction cost proportional to the size of the hedge trade. Teaches the agent to hedge infrequently and only when Greek exposure justifies it.

**Inventory risk:**
$$C^{\text{inventory}} = \lambda_\Delta \Delta_t^2 + \lambda_\Gamma |\Gamma_t| \cdot \sigma_t^2$$

Quadratic delta penalty (nonlinear risk of large positions) plus gamma penalty scaled by realized variance (gamma risk only materializes when markets move). $\lambda_\Delta, \lambda_\Gamma$ are risk aversion weights.

**Adverse selection cost:**
$$C^{\text{adverse}} = f_t^b \cdot \max(S_{t-1} - S_t, 0) \cdot q + f_t^a \cdot \max(S_t - S_{t-1}, 0) \cdot q$$

Penalizes fills that are immediately followed by adverse price moves — the signature of informed flow. Teaches the agent to widen quotes or reduce exposure when $\lambda_t^{\text{inf}}$ is elevated.

**Theoretical grounding:**
- $C^{\text{hedge}}$: Almgren-Chriss (2001) optimal execution
- $C^{\text{inventory}}$: Avellaneda-Stoikov (2008) market making
- $C^{\text{adverse}}$: Glosten-Milgrom (1985), Kyle (1985) adverse selection theory

---

### Discount Factor $\gamma$

$$\gamma \in [0.99,\ 0.999]$$

High $\gamma$ is justified because market making edge is cumulative — the agent must learn to manage inventory *now* to avoid losses *later*. The natural finite horizon imposed by $\tau$ provides additional structure: the agent knows expiry is coming and learns to manage Greeks more aggressively as $\tau \rightarrow 0$. At expiry the problem resets across option cycles.

---

## Belief Updater

The partially observable components require a belief updater over:

$$b_t = P(m_t, \sigma_t, r_t, \lambda_t^{\text{inf}} \mid h_t)$$

**Architecture: Rao-Blackwellized Particle Filter (RBPF)**

- **Particles** track the discrete regime variable $r_t$ — no distributional assumptions
- **Nonparametric filter** tracks continuous components $(\sigma_t, m_t, \lambda_t^{\text{inf}})$ conditioned on each particle's regime
- **Hawkes posterior** updates $\lambda_t^{\text{inf}}$ from fill event observations at each step
- No Gaussian assumptions anywhere — motivated by the well-documented fat tails and non-Gaussian behavior of financial markets

The RBPF is preferred over the Hamilton filter (which would handle regimes analytically) because the Hamilton filter requires Gaussian within-regime dynamics — an assumption that is empirically violated in crypto and equity markets, especially in the tails where risk management matters most.

---

## Key Tradeoffs the Agent Must Learn

| Tension | Description |
|---|---|
| Tight quotes vs. adverse selection | Tighter spreads fill more but attract informed flow |
| Frequent hedging vs. transaction costs | Delta neutrality is ideal but every hedge trade costs money |
| Short gamma vs. long theta | Holding inventory earns theta but bleeds on large moves |
| Aggressive quoting vs. inventory risk | More fills means more Greek exposure to manage |
| Information gathering vs. exploitation | Observing fills reveals regime information but at the cost of adverse selection exposure |

---

## Theoretical Foundations

| Model | Role in Formulation |
|---|---|
| Avellaneda-Stoikov (2008) | Baseline market making framework; motivation for quadratic inventory penalty |
| Glosten-Milgrom (1985) | Adverse selection in market making; informed vs. uninformed flow |
| Kyle (1985) | Price impact and informed trading |
| Almgren-Chriss (2001) | Optimal execution and transaction cost modeling |
| Hawkes (1971) | Self-exciting point process for order flow clustering |
| Stochastic Volatility models | Latent vol as a hidden state with its own noise process |
| RBPF (Rao-Blackwell) | Belief updater that exploits state structure without Gaussian assumptions |
