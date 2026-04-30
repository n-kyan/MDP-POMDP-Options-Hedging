# Options market making: a quant researcher's reference

Options market making (OMM) is best understood as a **stochastic optimal control problem under partial observation, executed on a hostile microstructure**, where the controller posts two-sided quotes on hundreds to thousands of contracts simultaneously, accumulates inventory whose risk lives on a high-dimensional Greek manifold, and hedges that risk in a market that punishes both action (transaction costs, signaling) and inaction (directional variance, pin risk). Profit comes from **three stacked sources**: bid-ask spread capture, the variance risk premium (implied > realized vol on average in SPX), and rebate/flow economics — not from directional bets on the underlying or vol. Everything else in this document is an elaboration of how to extract those three P&L streams without being destroyed by the fourth force: adverse selection.

---

## 0. Correcting and extending your mental model

Your framing is directionally correct and captures the core tensions better than most first-pass descriptions. Let me sharpen it before building out.

**What you got right.** The MM sets quotes around a fair-value estimate; Black-Scholes is a starting point but inadequate; inventory creates directional risk; flow segments into informed and uninformed; profit per trade depends on the distance between quote and truth; spread width trades fill rate against per-trade edge. These are the right load-bearing ideas.

**Six corrections that matter.**

1. **It's fair value, not "true mid," that defines edge.** There is no observable "true mid-price" the MM is reasoning about. The MM maintains a *conditional expectation* of the option's fair value $\hat{V}_t = \mathbb{E}[V_t \mid \mathcal{F}_t^{MM}]$ given their information filtration — quotes across strikes, order flow, the underlying tape, correlated instruments, vol-surface dynamics. Edge per trade is $|\text{quote} - \hat{V}|$ if $\hat{V}$ is correct, but the Bayesian point is that **$\hat{V}$ is itself biased by the flow you just saw**. When you get lifted on your ask, that lift *is information* that $\hat{V}$ was probably too low. Glosten-Milgrom formalizes this: the bid-ask spread in a zero-profit informed-trader equilibrium is exactly twice the expected update to fair value conditional on the trade direction. So the "distance to true mid" framing collapses into a self-referential update: your fair value *after* the trade is not your fair value *before* the trade.

2. **You cannot "avoid" informed traders — you price them.** An MM who refuses to trade with informed flow quotes infinite spreads and makes zero. The correct posture is (a) *charge* informed traders via the spread (Glosten-Milgrom compensation), (b) *detect* toxicity in real time and asymmetrically widen / fade on the side likely to be picked off, (c) *segment* flow by venue and counterparty (PFOF-purchased retail flow is cheap because it's uninformed), and (d) *use priority tools* — minimum quantity, post-only, auction participation — to avoid being the liquidity of last resort for latency arbitrageurs.

3. **Delta is the least interesting Greek for a real OMM book.** Delta is hedged quasi-continuously and cheaply via futures/ETFs; it's a solved problem in expectation. The risks that actually determine P&L are **vega bucketed by tenor and strike**, **gamma (especially cross-gamma with vol)**, and the cross-Greeks — **vanna** (skew exposure), **volga** (vol-of-vol / wing convexity), **charm** (time decay of delta; dominates near expiry and drives pin dynamics), plus higher orders **speed, zomma, color**. A flat-delta book with concentrated short vega in 30-day 90% strikes is not hedged; it's a short-skew trade. Institutional OMMs run risk on a multi-dimensional vega grid, not a scalar delta.

4. **The Black-Scholes objection is more surgical than "Gaussian returns are wrong."** BS fails in specific ways that each motivate a specific model extension: (a) constant vol → Heston/SABR/rough vol; (b) continuous paths → Merton/Bates jump diffusions; (c) lognormal marginals → local vol (Dupire) to match the entire observed surface; (d) independent increments → vol clustering / long memory → rough vol with Hurst $H \approx 0.1$. Importantly, the industry doesn't use one model — it uses **BS-in-IV-space** for communication and hedging, a parametric surface (SVI) for arbitrage-free interpolation, and a stochastic/local-vol hybrid for exotics and forward-start risk.

5. **Your spread-optimization intuition is exactly Avellaneda-Stoikov — but A-S makes it precise.** The formal model ties spread width to inventory $q$, time-to-horizon $T-t$, fill intensity parameters $(A, k)$, vol $\sigma$, and a risk-aversion $\gamma$. Critically, A-S produces *two* outputs: an optimal spread (symmetric, depends on $k, \gamma, T-t$) *and* a reservation-price shift (asymmetric, depends on $q, \gamma, \sigma^2, T-t$). The reservation shift is how inventory makes you skew: long inventory → you quote lower on both sides to attract sellers to you and repel buyers. A-S is a toy model (single asset, diffusion, no jumps, no adverse selection) but its decomposition — *symmetric spread for uncertainty, asymmetric skew for inventory* — is universal across every real OMM system.

6. **SPX European exercise simplifies PDE boundary conditions but complicates 0DTE.** No early exercise means no American premium, no exercise-boundary tracking, clean put-call parity. But European + cash-settled + AM settlement (SPX) creates **SOQ settlement risk** (you're pinned to an opening print you can't trade around), and 0DTE SPXW creates **intraday gamma cliffs** where an ATM 0DTE option's gamma diverges like $1/\sqrt{T-t}$ as $T-t \to 0$, producing hedging demands that the MM community now moves tens of billions of dollar-delta to absorb daily.

With those corrections in place, the rest of the document builds the full picture.

---

## 1. The problem formulation

### 1.1 OMM as stochastic optimal control

Consider an MM quoting a universe of $N$ options $\{O_i\}_{i=1}^N$ on a single underlying $S$. The state at time $t$:

$$
x_t = \big(S_t, \; \Sigma_t, \; q_t, \; \theta_t, \; \mathcal{L}_t, \; z_t\big)
$$

where $S_t$ is the underlying, $\Sigma_t$ is the implied vol surface (a function $\Sigma: (K, T) \mapsto \mathbb{R}_+$, in practice a finite parameter vector), $q_t \in \mathbb{Z}^N$ is per-contract inventory, $\theta_t$ is the aggregate Greek vector ($\Delta, \Gamma, \mathcal{V}, \Theta, \ldots$), $\mathcal{L}_t$ summarizes the limit order book (depth, imbalance, queue position for each of the MM's resting orders across exchanges), and $z_t$ is a latent regime / informed-trader indicator.

Controls: the MM chooses quotes $(\delta^b_{i,t}, \delta^a_{i,t})$ — bid/ask *offsets* from fair value — for each option, plus a hedge control $u^h_t$ (underlying shares traded, hedging options bought/sold).

Dynamics: underlying $dS_t = \mu\, dt + \sigma(S,t,\Sigma)\, dW_t + J\, dN_t$; vol surface $d\Sigma_t = \alpha_\Sigma\, dt + \beta_\Sigma\, dB_t$ (correlated with $W$); fills arrive as point processes with intensities $\lambda^{a,b}_i(\delta)$ that depend on how competitive the quote is; inventory jumps by $\pm 1$ on fills.

Objective (finite horizon $T$, e.g., end of day):

$$
V(x_0) = \sup_{\{\delta, u^h\}} \mathbb{E}\Big[U(X_T) \;-\; \int_0^T \phi(q_s, \theta_s)\, ds \;-\; \text{TxCost}\Big]
$$

where $U$ is terminal utility of cash, $X_T$ is mark-to-market wealth, and $\phi$ is a running inventory/risk penalty (typically a quadratic form on the Greek vector: $\phi = \tfrac{1}{2}\theta^\top \Lambda \theta$).

The Hamilton-Jacobi-Bellman equation is then (schematically)

$$
\partial_t V + \sup_{\delta, u^h}\Big\{ \mathcal{L}^S V + \sum_i \lambda^a_i(\delta^a_i)[V(q_i-1, \text{cash}+\text{ask}_i) - V] + \lambda^b_i[\cdots] - \phi \Big\} = 0
$$

with $\mathcal{L}^S$ the generator of price/vol dynamics. This is the general form; A-S is the simplest tractable special case.

### 1.2 POMDP framing

What's observable vs. hidden matters because the classical stochastic-control solutions assume full state observation.

| Observable | Hidden (latent) |
|---|---|
| NBBO, trades, own fills, volume, OPRA tape | True fair value $V_t^*$ |
| Underlying tape, futures, ETF | Informed-trader presence $z_t$ |
| Your inventory, Greeks, P&L | Vol regime (low-vol vs. crisis) |
| Resting quote priority (inferable) | Other MMs' inventories |
| Auction messages | Upcoming news / macro shocks |

The POMDP structure is essential: optimal policy is a function of the **belief state** $b_t = p(z_t, V^*_t \mid \mathcal{F}_t^{MM})$, not the observation directly. Concretely, after a large buy print at your ask, you update belief toward "informed buyer present" and widen/skew before the next tick — this is *exactly* the Glosten-Milgrom Bayesian update embedded inside a control loop. RL for market making is appealing precisely because POMDPs are generally intractable analytically, and modern actor-critic methods can learn good belief-conditioned policies from simulation.

### 1.3 The core tradeoffs

- **Spread width vs. fill rate**: wider spreads capture more per fill but reduce fill intensity $\lambda(\delta) = A e^{-k\delta}$ (A-S-style exponential fill model).
- **Inventory vs. hedge cost**: carry inventory and pay risk (quadratic penalty in $q$), or hedge and pay transaction cost (linear in $|u^h|$ plus quadratic market impact).
- **Speed vs. thoughtfulness**: faster quote updates reduce stale-quote picking but cost infrastructure and signal strategy to observers.
- **Model complexity vs. robustness**: richer models fit better in-sample, overfit out-of-sample, and add latency.

---

## 2. Fair value estimation

### 2.1 From Black-Scholes to the industry's actual stack

Black-Scholes is not how OMMs price. It is the **coordinate system** in which they communicate. Given a quote price $C^{mkt}$, inverting BS yields implied vol $\sigma^{IV}(K,T)$. Prices are quoted, risk-managed, and interpolated in IV space because IV is a much smoother and more stationary object than price.

BS assumes $dS/S = \mu\, dt + \sigma\, dW$ with constant $\sigma$, giving

$$
C^{BS}(S, K, T, \sigma, r) = S\, N(d_1) - K e^{-rT} N(d_2), \quad d_{1,2} = \frac{\ln(S/K) + (r \pm \sigma^2/2)T}{\sigma\sqrt{T}}.
$$

Empirical failures, each with a precise fix:

| Failure | Evidence | Model fix |
|---|---|---|
| Constant vol | Smile/skew across $K$; term structure across $T$ | Local vol (Dupire); stochastic vol (Heston, SABR); rough vol |
| No jumps | Overnight gaps, earnings, macro shocks; fat-tailed returns (kurtosis 5–30 vs. 3) | Merton jump-diffusion; Bates (Heston + jumps); Lévy processes |
| Lognormal marginals | SPX 30-day returns have skewness $\approx -1$ | Variance gamma; NIG; rough Bergomi |
| Independent increments | Vol clustering; ACF of $|r_t|$ decays as power law | GARCH; rough vol with $H \approx 0.1$ |
| Constant rates | Rates are stochastic, matter for long-dated | Hull-White; SABR-LMM for rates+vol |

### 2.2 The volatility surface

The object of interest is $\Sigma(K, T)$, or equivalently in *moneyness* $k = \ln(K/F)$ where $F = S e^{(r-q)T}$ is the forward, and total implied variance $w(k,T) = \Sigma^2(k,T) T$.

**Stylized facts for SPX:**
- Persistent negative skew: OTM puts trade at higher IV than OTM calls. Two explanations: the **leverage effect** (Black 1976: equity vol rises as price falls because leverage increases) and **crash-o-phobia** (Rubinstein 1994: post-1987 risk premium on downside tail). Empirically the skew is steeper than leverage alone predicts, so crash-o-phobia dominates at the wings.
- Term structure: usually in **contango** (longer tenors = higher IV) in calm regimes; inverts to **backwardation** in crises (VIX > VIX3M).
- ATM skew term structure: $|\partial_k \Sigma(0, T)| \sim T^{-H-1/2}$ with $H \approx 0.1$ — the *fingerprint of rough vol* (Gatheral-Jacquier-Rosenbaum 2018).

**SVI parameterization (Gatheral 2004):** the raw SVI slice has total variance

$$
w(k) = a + b\big(\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\big)
$$

Five parameters per slice, asymptotically linear wings (matches Lee's moment formula: $w(k)/|k| \to \beta_\pm$ where $\beta_\pm \le 2$ for no-arbitrage). **Gatheral-Jacquier (2014)** prove that **SSVI** (surface SVI) with a single correlation $\rho$ and a power-law ATM variance curve is arbitrage-free under explicit parameter conditions — this is the workhorse for production surface fitting because it handles both calendar and butterfly no-arbitrage by construction.

No-arbitrage constraints are not negotiable:
- **Butterfly no-arb (strike-convexity)**: $\partial^2 C / \partial K^2 \ge 0$, equivalently the risk-neutral density $\phi(K) = e^{rT}\partial^2 C/\partial K^2 \ge 0$.
- **Calendar no-arb**: $w(k, T)$ non-decreasing in $T$ for each $k$.
- **Lee's moment formula**: wing behavior bounds on IV as $|k| \to \infty$.

A surface that violates either admits a static arbitrage — an MM who quotes through a violation gets run over by stat-arb desks instantly.

### 2.3 Local vs. stochastic vs. rough vol

**Dupire local vol (1994)**: given an arbitrage-free call surface $C(K,T)$, there exists a unique local vol function $\sigma_{LV}(K,T)$ such that $dS_t/S_t = \sigma_{LV}(S_t, t)\, dW_t$ reproduces the surface exactly:

$$
\sigma_{LV}^2(K,T) = \frac{\partial_T C + (r-q) K \partial_K C + qC}{\tfrac{1}{2} K^2 \partial^2_{KK} C}.
$$

Local vol is a *fitting* tool, not a dynamic one. It reprices vanillas by construction but gets **forward smile dynamics** wrong — the skew flattens as spot moves, which contradicts the **sticky-strike / sticky-delta** behavior observed. Use for exotics pricing with caution.

**Heston (1993)**: stochastic variance

$$
dv_t = \kappa(\theta - v_t)\, dt + \xi\sqrt{v_t}\, dZ_t, \quad d\langle W, Z\rangle_t = \rho\, dt.
$$

Produces skew via $\rho < 0$ and smile via $\xi$. Semi-closed-form via characteristic function + FFT (Carr-Madan 1999). Fails to produce short-dated skew steep enough to match SPX — because $\sigma_{ATM}$'s skew decays like $1/T$ in any diffusive stoch-vol model, but empirically decays like $T^{-0.4}$.

**SABR (Hagan et al. 2002)**: $dF = \alpha F^\beta dW$, $d\alpha = \nu \alpha\, dZ$. Has a celebrated asymptotic IV formula used everywhere for single-strike interpolation, especially in rates. Wing behavior can produce arbitrage and requires care.

**Rough vol (Gatheral-Jaisson-Rosenbaum 2018)**: log-volatility is a fractional Brownian motion with $H \approx 0.1$:

$$
\log \sigma_t = \log \sigma_0 + \eta\, W_t^H.
$$

**Rough Bergomi** is the pricing workhorse: forward variance curves driven by a Volterra process. Matches the empirical $T^{-H-1/2}$ ATM skew decay. Computationally expensive (non-Markovian); Markovian approximations (Abi Jaber-El Euch 2019; Bayer et al.) are the current practical compromise. As of 2024–2026, rough vol is in production at top firms for surface calibration and short-dated skew but not yet for desk-wide risk engines, where Markovian stoch-vol + jumps still dominate.

**Jumps — Merton & Bates**: Merton (1976) adds compound Poisson jumps to GBM; Bates (1996) adds them to Heston. For OMMs, jump models matter because **jump risk is unhedgeable with continuous delta** — the residual gamma term $\tfrac{1}{2}\Gamma J^2$ appears on every jump. The market prices a jump risk premium (visible as short-dated OTM put skew steeper than diffusive models can produce), and a sophisticated OMM decomposes vega exposure into diffusive vega and jump-risk vega.

### 2.4 From model to MM fair value

In practice an OMM fair-value stack looks like:
1. Pull raw NBBO quotes across all listed strikes/tenors.
2. Filter (crossed, locked, stale).
3. Back out IV via Black-76 (futures-style, since SPX is European on forward).
4. Fit a parametric surface (SSVI or similar) with no-arb constraints.
5. Blend in microstructure signals: order book imbalance on underlying, futures basis, ETF-index arb signals, VIX/VVIX moves, recent trade flow (sign-adjusted).
6. Produce a *fair value per strike* $\hat{V}_i$ that is internally consistent across the book and updated on every relevant tick.

Cross-asset signals are not optional. SPX fair value depends on ES futures basis, SPY-index arb, VIX term structure (for vega by tenor), sector dispersion (for correlation-sensitive wings), and rate curves (for $r-q$ forward). Single-name options additionally require discrete dividend schedules, hard-to-borrow rates (which enter as an effective $q$), and earnings-date vol bumps — most naïve surfaces badly misprice single-name options through earnings without a term-structure kink at the event date.

---

## 3. Quote setting

### 3.1 The Avellaneda-Stoikov derivation

Set up: a single asset with mid $S_t$ following $dS_t = \sigma\, dW_t$. An MM holds cash $X_t$ and inventory $q_t$ (shares, can be negative). MM posts bid $S_t - \delta^b$ and ask $S_t + \delta^a$. Fill intensities are

$$
\lambda^a(\delta^a) = A e^{-k\delta^a}, \qquad \lambda^b(\delta^b) = A e^{-k\delta^b},
$$

so fills follow Poisson processes with controllable rates. On a buy fill (bid hit): $q \to q+1$, $X \to X - (S - \delta^b)$. On sell fill: $q \to q-1$, $X \to X + (S + \delta^a)$.

Objective: maximize exponential utility of terminal wealth,

$$
u(s, x, q, t) = \sup_{\delta^a, \delta^b} \mathbb{E}\big[-\exp(-\gamma(X_T + q_T S_T))\big].
$$

**Step 1 — HJB.** Applying the dynamic programming principle,

$$
\partial_t u + \tfrac{1}{2}\sigma^2 \partial^2_{ss} u + \sup_{\delta^a}\lambda^a(\delta^a)[u(s,x+s+\delta^a, q-1, t) - u] + \sup_{\delta^b}\lambda^b(\delta^b)[u(s,x-s+\delta^b, q+1, t) - u] = 0.
$$

**Step 2 — ansatz.** Try $u(s,x,q,t) = -\exp(-\gamma x)\exp(-\gamma q s)\exp(-\gamma \theta(q,t))$, reducing to a PDE in $\theta$.

**Step 3 — solve reservation price first (no quotes).** If the MM cannot trade, the indifference price at which they are willing to buy one more share is the **reservation price**:

$$
\boxed{\; r(s, q, t) = s - q\,\gamma\,\sigma^2\,(T-t). \;}
$$

Long inventory $(q>0)$ shifts $r$ below $s$ — the MM values the asset less than the market because holding more of it adds variance to terminal wealth.

**Step 4 — optimal quotes around $r$.** Plugging back and optimizing the intensity FOCs,

$$
\delta^{a*} - (r - s) = \frac{1}{\gamma}\ln\!\left(1 + \frac{\gamma}{k}\right), \qquad (s - r) - \delta^{b*} = \frac{1}{\gamma}\ln\!\left(1 + \frac{\gamma}{k}\right) \; \text{(sign corrected)}.
$$

More cleanly, the **total optimal spread** (symmetric around $r$) is

$$
\boxed{\; \delta^{a*} + \delta^{b*} = \gamma\,\sigma^2\,(T-t) \;+\; \frac{2}{\gamma}\ln\!\left(1 + \frac{\gamma}{k}\right). \;}
$$

The spread has **two components**: an inventory/risk-aversion piece $\gamma\sigma^2(T-t)$ (wider when vol is high, horizon long, or risk aversion high) and a microstructure piece $(2/\gamma)\ln(1+\gamma/k)$ (wider when fill intensity is low, i.e., adverse selection is high or competition is soft).

**Interpretation.** A-S decomposes quoting into two orthogonal decisions: (i) **where to center quotes** (reservation price, depends on inventory), (ii) **how wide to make them** (depends on volatility, horizon, and fill-intensity parameters). This decomposition is universal — every real OMM system has these two knobs, even when the underlying model is far richer.

### 3.2 Worked numerical example

SPX options context. Suppose an MM quotes one specific option with: vol of the option's mid-price $\sigma = \$0.40$/minute, risk aversion $\gamma = 0.1$, fill intensity $A = 1.5$/min at $\delta=0$, decay $k = 1.5$/\$, horizon $T-t = 60$ min, current inventory $q = +5$ contracts.

- Reservation shift: $q\gamma\sigma^2(T-t) = 5 \cdot 0.1 \cdot 0.16 \cdot 60 = \$4.80$. So $r = s - 4.80$: long inventory pushes both quotes down by \$4.80 (per contract-price units; in practice this would be scaled).
- Inventory spread component: $\gamma\sigma^2(T-t) = 0.1 \cdot 0.16 \cdot 60 = \$0.96$.
- Microstructure component: $(2/0.1)\ln(1 + 0.1/1.5) = 20 \cdot \ln(1.0667) = 20 \cdot 0.0645 = \$1.29$.
- Total spread: $\$2.25$; half-spread $\$1.125$ around $r$.
- Final quotes: $\text{bid} = s - 4.80 - 1.125 = s - 5.925$; $\text{ask} = s - 4.80 + 1.125 = s - 3.675$.

Both quotes are below $s$ — the long inventory has skewed the MM into actively trying to sell.

### 3.3 Ho-Stoll (1981)

The predecessor to A-S: an inventory-based model where MMs maximize profit subject to a quadratic inventory cost. Core insight — **spreads widen and skew with inventory** — is the same as A-S. The difference is Ho-Stoll doesn't derive quotes from a fill-intensity model, so the microstructure (spread-width) component is imposed rather than derived. Historically important; A-S subsumes it.

### 3.4 Glosten-Milgrom (1985)

A different foundation: the spread as pure adverse-selection compensation. Setup: informed traders know the true value $V \in \{V_L, V_H\}$; uninformed trade for exogenous reasons. A fraction $\alpha$ of traders are informed. The MM (zero-profit, Bayesian) sets

$$
\text{ask} = \mathbb{E}[V \mid \text{buy}], \qquad \text{bid} = \mathbb{E}[V \mid \text{sell}].
$$

The spread is exactly the information asymmetry: $\text{ask} - \text{bid} = \mathbb{E}[V|\text{buy}] - \mathbb{E}[V|\text{sell}]$, which grows with $\alpha$ and with the spread $V_H - V_L$ of possible true values. **Key insight for OMMs**: every trade is a Bayesian signal about $V$. Post-trade, update $\hat{V}$, then re-quote.

In options context this becomes: a large buy of 30-delta puts updates $\hat{V}$ for *every option correlated with that one* — nearby strikes, nearby tenors, correlated names — because the trade reveals either (a) directional view on the underlying (shifts entire surface via spot) or (b) vol view (shifts the surface via $\Sigma$). Modern OMMs propagate the Bayesian update across the whole book, not just the traded contract.

### 3.5 Kyle (1985)

A single informed trader with private signal $v \sim N(p_0, \Sigma_0)$ trades a quantity $x$; noise traders add $u \sim N(0, \sigma_u^2)$; MM sees only total $y = x + u$ and sets price linearly $p = p_0 + \lambda y$. Equilibrium: the informed trader trades $x = \beta(v - p_0)$ and the MM's optimal price impact is

$$
\boxed{\;\lambda = \frac{1}{2}\sqrt{\Sigma_0 / \sigma_u^2}.\;}
$$

**Kyle's lambda** is the empirical measure of price impact per unit volume. In options, $\lambda$ is tenor- and strike-specific and substantially higher than in the underlying: an informed trade in 30-delta SPX puts moves the *entire put skew*, not just that strike.

### 3.6 Cartea-Jaimungal and the modern extensions

The modern literature (Cartea-Jaimungal-Penalva, *Algorithmic and High-Frequency Trading*, 2015, plus the RL extensions through 2024) extends A-S along several axes that matter in practice:

- **Order flow signals**: adding a stochastic drift term for $S$ driven by order book imbalance. The reservation price picks up an alpha term beyond the inventory term.
- **Adverse selection**: modeling fills as *informed* with probability $\pi_t$ that is itself state-dependent. Optimal policy widens the side likely to be picked.
- **Multiple assets / options surface**: inventory becomes a vector $q \in \mathbb{R}^N$, risk penalty becomes $\tfrac{1}{2}q^\top \Lambda q$ with $\Lambda$ the *Greek risk matrix*, and quotes are coupled across contracts because a fill in one changes the optimal quotes in correlated contracts.
- **Market impact on hedge**: including a cost for hedging trades, yielding the joint quote-and-hedge policy. Cartea-Sánchez-Betancourt (2021) and follow-ups formalize this.

For practical implementation, the 2024 literature (Coache-Cartea-Jaimungal on conditionally elicitable risk measures, Buehler-Murray-Wood on Deep Bellman Hedging) replaces the analytic HJB solution with deep RL and neural hedging — same problem, more flexible function classes.

### 3.7 Quoting mechanics in practice

Beyond the optimal $(\delta^a, \delta^b)$:
- **Multi-level quoting**: rest at multiple price levels to capture different fill rates vs. edge.
- **Asymmetric widening on toxicity**: on detecting informed flow (large print, correlated instruments moving), widen the vulnerable side first while holding the safe side.
- **Quote fading**: pull quotes entirely when toxicity exceeds threshold — accept zero fill rate to avoid negative expected edge.
- **Priority filters**: minimum quantity, post-only, participate-only-in-auction flags to filter order types.
- **Inventory-driven skew**: the A-S reservation-price shift, extended to Greek inventory: if the book is short vega in a specific bucket, skew all quotes in that bucket to buy vega back.

---

## 4. Inventory and risk management

### 4.1 The full Greek taxonomy

| Greek | Definition | Why it matters for OMM |
|---|---|---|
| Delta $\Delta = \partial V/\partial S$ | Spot sensitivity | First-order directional; hedged continuously via futures/ETF |
| Gamma $\Gamma = \partial^2 V/\partial S^2$ | Delta convexity | P&L on realized vs. implied vol; hedge cost |
| Vega $\mathcal{V} = \partial V/\partial \sigma$ | Vol sensitivity | Primary exposure for OMM books; bucketed by tenor/strike |
| Theta $\Theta = \partial V/\partial t$ | Time decay | Paid to gamma holder; the "carry" of long-option positions |
| Rho $\rho = \partial V/\partial r$ | Rate sensitivity | Matters for long-dated; small for short-dated |
| Vanna $\partial^2 V/\partial S\partial\sigma$ | Delta's vol sensitivity (or vega's spot sensitivity) | **Skew exposure**; how delta hedge moves as vol moves |
| Volga/vomma $\partial^2 V/\partial\sigma^2$ | Vega convexity | **Wing risk**; OTM options have high volga |
| Charm $\partial^2 V/\partial S\partial t$ | Delta's time decay | Dominates near expiry; drives pin dynamics |
| Speed $\partial^3 V/\partial S^3$ | Gamma's spot sensitivity | Gamma stability under large moves |
| Zomma $\partial^3 V/\partial S^2\partial\sigma$ | Gamma's vol sensitivity | Gamma hedge stability under vol moves |
| Color $\partial^3 V/\partial S^2 \partial t$ | Gamma's time decay | Intraday gamma re-estimation |

A professional OMM runs *Greek ladders*: a table of aggregate exposures by (strike bucket, tenor bucket) for each Greek, not just the scalar book totals. A book that's scalar-flat vega can have massive short-skew via vanna concentrated in short-dated wings.

### 4.2 Vega bucketing

Vega is not a scalar because the vol surface is a surface. The correct representation is **bucketed vega**: $\mathcal{V}_{i,j} = \partial V / \partial \Sigma(K_i, T_j)$ for a grid of $(K,T)$ nodes. A book can be vega-neutral in aggregate but long 30-day 90% vega and short 180-day ATM vega — a massive *term-structure and skew* trade.

Risk limits operate on this grid: hard caps on each node, a total L2 norm cap, and *scenario* caps (e.g., cap on P&L from a parallel +5 vol shift combined with -5% spot).

### 4.3 VaR, ES, and stress testing

VaR and ES are computed via historical or Monte Carlo simulation with full surface re-pricing — options books are non-linear enough that delta-normal VaR badly understates risk. Stress tests required at major firms:

- **Spot shocks**: ±5%, ±10%, ±20%.
- **Vol shocks**: parallel ±5 vol pts, skew twist (flatten/steepen by 20%).
- **Combined**: spot down 10% *with* vol up 10 pts (the 1987/2008/2020/Aug-2024 archetype).
- **Pin risk**: scenarios where spot pins to a large-OI strike at expiry.
- **Gamma squeeze**: large directional move with dealer-hedging amplification.

Aug 5, 2024 (VIX intraday ~65 on yen carry unwind) and April 4, 2025 ("Liberation Day," VIX close 45.3) both produced widespread stress-scenario triggers across OMM risk systems, tightening vega/gamma limits across the industry.

### 4.4 Non-Gaussian returns and hedging incompleteness

Empirical SPX daily log-returns have kurtosis ~10–30 (vs. Gaussian 3) and skewness $\approx -1$. Implications:

- Delta hedging errors have fat tails: $\mathbb{E}[\text{hedge error}^2]$ is finite but $\mathbb{E}[|\text{hedge error}|^p]$ for $p > 2$ blows up quickly.
- Large moves produce **gamma P&L** that BS attributes to "vol was wrong" but is really **jump risk realizing**.
- Markets are **incomplete** in the presence of jumps: no replicating portfolio exists for vanilla options using only the underlying. This is the theoretical justification for the variance risk premium — OMMs earn it because they're bearing unhedgeable jump risk.

### 4.5 The variance risk premium

Implied vol is systematically higher than realized vol. For SPX 30-day: long-term average VIX $\approx$ 19, long-term realized $\approx$ 16. The difference (~3 vol pts, or ~30% of realized) is the **variance risk premium** (VRP), paid by option buyers to option sellers as compensation for bearing jump/crash risk. OMMs implicitly harvest this because their inventory, averaged over time, tilts short vol (buyers of protection dominate flow, so MMs end up short). This is a *structural* P&L source that's reliable in normal regimes and catastrophic in left-tail events.

---

## 5. Hedging

### 5.1 Delta hedging: continuous vs. discrete

Under BS assumptions, continuous delta hedging replicates perfectly. In reality:

- **Discrete rehedging** introduces tracking error. If you rehedge at intervals $\Delta t$, the variance of hedging P&L scales as $\sigma^2 \Gamma^2 S^4 \Delta t$.
- **Transaction costs** (Leland 1985): rehedging every $\Delta t$ with proportional cost $\kappa$ is equivalent to trading a modified BS with vol $\tilde\sigma^2 = \sigma^2(1 + \kappa\sqrt{8/(\pi\sigma^2\Delta t)})$. Cost scales as $1/\sqrt{\Delta t}$ — rehedge too often and costs explode.
- **Whalley-Wilmott bands**: optimal rehedge when $|\text{delta error}| > \text{band}$, where band $\propto (\text{cost}\cdot \Gamma^2 S^2/\gamma)^{1/3}$. This is the standard for cost-aware discrete delta hedging.

Modern OMMs don't rehedge per option — they compute net book delta, net it against existing hedges, and rehedge the net at some frequency (typically seconds to minutes in normal regimes, tick-level in stress).

### 5.2 Gamma, vega, and cross-Greek hedging

- **Gamma hedging** requires options (can't hedge convexity with linear instruments). Typically done with liquid ATM options (highest gamma per dollar vega).
- **Vega hedging** uses vega-bucketed portfolios: sell vega in overbought buckets, buy in undersold ones. Cross-hedging across tenors via the vol term structure (VIX futures for short tenors, VIX options for vol-of-vol).
- **Vanna/volga hedging**: needed when book has concentrated skew risk. The vanna-volga method (Castagna-Mercurio 2007) prices exotics by adjusting for vanna/volga costs using market-observed vanilla prices.

### 5.3 Semi-static hedging: Carr-Madan

**Carr-Madan (2001)** replication formula: any twice-differentiable payoff $f(S_T)$ can be statically replicated by a combination of a bond, forward, and a continuum of vanilla puts/calls:

$$
f(S_T) = f(\kappa) + f'(\kappa)(S_T - \kappa) + \int_0^\kappa f''(K)(K-S_T)^+ dK + \int_\kappa^\infty f''(K)(S_T-K)^+ dK.
$$

This is how **variance swaps** are replicated: $\log(S_T/S_0)$ replicates to a strip of OTM options weighted by $1/K^2$. The VIX is literally computed from this formula applied to the 30-day SPX option strip. For OMMs, Carr-Madan means a large class of exotic risks can be *statically* hedged with vanillas at inception, eliminating dynamic-hedging errors — at the cost of needing the full strip of vanillas liquidly available.

### 5.4 When not to hedge

- Small positions relative to risk limits — transaction cost exceeds expected risk cost.
- Offsetting flow expected soon (e.g., opposite side of a known dispersion trade).
- Regime shifts where hedging would lock in losses (after a big move, wait for mean reversion).
- Positions where the hedge itself is toxic (hedging a thinly-traded name via correlated ETF introduces basis risk).

The decision is a cost-benefit: $\mathbb{E}[\text{risk cost}] \cdot \Delta t$ vs. transaction cost, compared at the margin.

### 5.5 Hedging in single-name vs. SPX

| Dimension | SPX | Single-name |
|---|---|---|
| Delta hedge | ES futures (deep, cheap, near-continuous) | Stock (variable liquidity, HTB risk) |
| Gamma hedge | Liquid ATM SPX/SPY options | Often only one reasonably liquid strike |
| Dividend risk | Smooth index dividend | Discrete, event-driven, ex-date jumps |
| Earnings jumps | N/A | Dominant risk; vol term structure has event-date kink |
| Borrow | Negligible | HTB stocks: borrow cost enters as effective dividend |
| Exercise | European | American: early exercise risk on ITM puts near dividends |
| Rehedge frequency | Sub-second | Seconds to minutes |

---

## 6. Adverse selection and order flow

### 6.1 Taxonomy

**Informed flow**: hedge funds with views, prop shops with signals, sophisticated retail (a small but growing fraction), index funds with forced rebalancing (informed by construction even if not by discretion).

**Uninformed flow**: retail speculation, corporate hedging, dealer unwinds, mechanical strategies (covered calls, collar rolls).

Retail options flow is valuable because it's (a) small per-order, (b) uncorrelated with future price moves at short horizons, (c) time-insensitive. This is why wholesalers like Citadel Securities and Virtu pay brokers PFOF for retail flow — it's a subsidized lunch.

### 6.2 Toxicity metrics

**PIN (Easley-Kiefer-O'Hara 1996)**: maximum-likelihood estimate of probability of informed trading from trade-direction imbalances. Slow (daily).

**VPIN (Easley-López de Prado-O'Hara 2012)**: volume-clock version. Computes imbalance between buy volume and sell volume over volume buckets:

$$
\text{VPIN} = \frac{\sum_{\tau=1}^n |V^B_\tau - V^S_\tau|}{\sum_{\tau=1}^n V_\tau}.
$$

High VPIN predicts adverse-selection episodes. Controversial (criticized for being a noisy proxy) but widely used as a toxicity indicator.

**Realized quote fade rates**: the fraction of your quotes hit within $N$ ms of a correlated price move — operational measure of how "stale" your quotes are.

### 6.3 Adverse selection specific to options

Signals of informed options flow:

- **Unusual size**: 2000-lot on an otherwise quiet name.
- **OTM / short-dated skew**: deep OTM short-dated calls on a name before an announcement.
- **Aggressive pricing**: willing to pay mid or cross the spread.
- **Cross-instrument confirmation**: simultaneous buying in correlated names or related futures.
- **Multi-leg precision**: structures that require view (risk reversals, ratio spreads) vs. retail defaults (single calls/puts, vertical spreads).

OMM response: widen the side likely to be picked, shrink size at your quote, propagate the Bayesian update across the surface, and if magnitude is large, fade quotes entirely.

### 6.4 PFOF and flow segmentation

As of 2025–2026: PFOF remains legal in the U.S.; the SEC's 2022 Order Competition Rule was **withdrawn in 2025** under the Atkins SEC. PFOF is prohibited in the UK/EU. The retail wholesaler concentration in options remains high: Citadel Securities self-reports ~30% of all U.S. listed options market share and is the largest retail options wholesaler; Virtu and G1X are the other top players.

Effect on OMMs: the internalized retail flow that reaches Citadel/Virtu is **prized** (uninformed, small, price-insensitive). Flow that reaches lit exchanges is disproportionately **informed** — because informed flow by definition cannot get filled at internalized midpoint prices without revealing itself. This creates a structural bifurcation: PFOF-buying wholesalers harvest the good flow; on-exchange MMs face adversely-selected residual.

---

## 7. Microstructure and execution

### 7.1 The U.S. options exchange landscape (2025-2026)

**18 U.S. options exchanges** across 5 operators:

| Operator | Exchanges |
|---|---|
| Cboe (4) | Cboe Options (C1), C2, BZX Options, EDGX Options |
| Nasdaq (6) | PHLX, NOM, BX Options, ISE, GEMX, MRX |
| NYSE (2) | NYSE American, NYSE Arca |
| MIAX (4) | MIAX Options, Pearl, Emerald, **Sapphire** (launched Aug 2024, physical floor 2025) |
| BOX (1) | BOX Options |
| MEMX (1) | **MEMX Options** (launched Sept 2023) |

Pending: **IEX Options** (with asymmetric speed bump + Options Risk Parameter, contested by Citadel/SIFMA) and **MEMX MX2** (customer-priority pro-rata, planned Q2 2026). Only 5 exchanges retain trading floors.

**Critical for SPX specifically**: SPX options trade *only on Cboe*. This is a monopoly venue. SPY (ETF equivalent, 1/10 notional ratio but American-style, physically settled) is multi-exchange and subject to cross-exchange competition.

### 7.2 Pricing models

- **Maker-taker**: pay rebate to resting liquidity, charge fee to aggressors. Favored by passive MMs.
- **Taker-maker** (inverted): charges resting, rebates aggressors. Favored by flow-seeking firms. MIAX Sapphire uses this.
- **Customer-priority / pro-rata**: customer orders get priority over MMs; among non-priority, allocation is pro-rata by size rather than time. Dominant on ISE and historically on Cboe for certain classes; rewards size commitment.
- **Price-time priority**: FIFO within price level. MEMX, most modern venues.

### 7.3 Auction mechanisms

- **AIM (Automated Improvement Mechanism)** on Cboe; similar **PIM/Facilitation** on ISE/PHLX. A broker submits a paired retail order + MM price; other MMs can price-improve in a ~1-second auction. Designed to benefit retail customers while giving the initiating MM priority. Effectively the main mechanism for PFOF-sourced flow to reach lit venues.
- **QCC (Qualified Contingent Cross)**: pre-negotiated multi-leg cross printed on-exchange without auction. Used for institutional block flow.
- **SAM (Solicitation Auction)** for large institutional orders.
- **Complex Order Book (COB)**: native handling of multi-leg orders (spreads, butterflies, condors). Liquidity in COB is often implied from single-leg books; complex orders can "leg out" against singles.
- **FLEX**: fully-customizable options (strike, expiry, exercise style). 2025 ADV $\approx$ 1.3M contracts, 10× 2019 level. Important for OMMs because FLEX flow carries less adverse-selection risk (bilateral institutional) but more model risk (non-standard terms).

### 7.4 Market-maker obligations

Registered MMs (and especially Lead Market Makers / Designated Primary MMs) have regulatory obligations: quote a specified percentage of series continuously, with max spread width, within a quoted time. In exchange: priority, rebate tiers, and sometimes allocation benefits. The obligation matters because it's a binding constraint on "pulling quotes" during toxicity — at some point you must keep quoting to retain status.

### 7.5 Queue position and latency

On price-time venues, queue position determines fill probability. First-in at a given price level fills first. Queue position is *earned* by posting early and *lost* by re-quoting, so there's a tension: updating fair value requires re-quoting, but re-quoting forfeits queue. The industry solution is incremental modification where allowed (some venues preserve time priority on size increases) and aggressive use of pro-rata venues where queue matters less.

Latency: SPX options are on Cboe only, so latency arbitrage across exchanges isn't the issue — the issue is *information arbitrage* against the ES futures tape and SPY. Stale SPX quotes can be picked off in microseconds after ES moves. Every serious SPX OMM co-locates at CME (ES) and Cboe with dedicated cross-connects and hardware timestamping.

### 7.6 SPX vs SPXW settlement

| Contract | Expiration | Style | Settlement | Last trade |
|---|---|---|---|---|
| SPX (3rd Friday monthly) | 3rd Friday | European | **AM-settled (SOQ)** | Thursday before |
| SPXW (weeklys, EOM, PM-settled 3rd Friday) | Any weekday | European | **PM-settled (4:00pm close)** | Day of expiry |
| XSP | Like SPXW | European | PM-settled | Day of expiry |
| SPY | 3rd Friday / weeklys | American | Physical | Day of expiry |

AM-settled SOQ is a computed opening print across all 500 constituents, each opening at different times in the first minutes of trade. MMs short SOQ-settled options face **settlement basis risk**: you can hedge the forward intraday but cannot trade the SOQ print itself. Bespoke hedging strategies (MOO basket orders on constituents) are used by the top desks.

---

## 8. SPX-specific considerations

### 8.1 The 0DTE revolution

The single most important structural change in SPX options since 2022 is **0DTE**: options expiring same-day. Trajectory:

| Period | 0DTE share of SPX volume |
|---|---|
| 2016 | ~5% |
| May 2022 | CBOE completes daily expirations (T/Th added) |
| End-2023 | ~45% |
| Full-year 2024 | ~47% |
| Aug 2025 | 62.4% (record monthly) |
| **Full-year 2025** | **59% of SPX volume; ~2.3M contracts ADV** |

Retail is ~50–60% of 0DTE flow; >95% of 0DTE trades are limited-risk (long options or debit/credit spreads). Total 2025 U.S. listed options volume: **15.2B contracts (+26% YoY, 6th consecutive record)**. Single-day record Oct 10, 2025: 110M contracts.

**Why 0DTE matters to OMMs**: gamma on a 0DTE ATM option goes like $1/\sqrt{T-t}$, so as $t \to T$ the dealer's gamma inventory (and therefore hedging demand) diverges. Academic evidence (Dim-Eraker-Vilkov 2024; Almeida-Freire-Hizmeri 2024; Cboe 2024 white paper) is that on average net dealer gamma in 0DTE is positive and *attenuates* intraday vol; but in stress (Aug 5 2024, April 2025) local dealer-short-gamma pockets around heavy strikes *amplify* moves. This is the new dominant intraday vol regime and has forced every SPX MM desk to rebuild its end-of-day hedging stack.

### 8.2 VIX, VVIX, and variance replication

VIX is computed as a weighted strip of 30-day OTM SPX options:

$$
\text{VIX}^2 = \frac{2e^{rT}}{T}\sum_i \frac{\Delta K_i}{K_i^2} Q(K_i) - \frac{1}{T}\Big(\frac{F}{K_0}-1\Big)^2
$$

— this is the Carr-Madan replication of $\log(S_T/F)$, hence the variance-swap interpretation. VVIX (vol-of-VIX) is the equivalent over VIX options. For OMMs, these indices are:
- Hedge instruments for vega (VIX futures) and volga (VIX options).
- Fair-value inputs: SPX vol surface must be consistent with VIX term structure to avoid arbitrage.
- Indicators of regime: VIX > 30 or VVIX > 120 signals vol regime shift.

### 8.3 Correlation risk and dispersion

SPX implied vol reflects not just constituent vols but *implied correlation*:

$$
\sigma^2_{\text{index}} \approx \sum_i w_i^2 \sigma_i^2 + \sum_{i\ne j} w_i w_j \rho_{ij} \sigma_i \sigma_j.
$$

**Dispersion trading** (short index vol, long basket of constituent vols) is a counterparty that systematically sells correlation to OMMs. An SPX MM must monitor implied correlation (derivable from index vol and constituent vols) because dispersion flow biases the index-single-name vol relationship.

---

## 9. What makes OMM difficult

The problem is hard because every source of difficulty compounds with every other.

- **The latent-fair-value problem** (Section 2): you never know your target; every trade shifts it.
- **High-dimensional risk**: hundreds of contracts, dozens of Greeks, bucketed by strike and tenor — the state space is genuinely huge.
- **Non-Gaussian, non-stationary dynamics**: jumps, regime shifts, rough vol, correlation breakdowns in stress.
- **Adverse selection**: every counterparty is potentially informed; you cannot distinguish ex ante.
- **Competition**: multiple MMs price the same option; winner's curse on every fill.
- **Latency arms race**: stale quotes get picked; infrastructure cost is $100M+ to play at the top tier.
- **Regulatory obligations**: can't pull quotes entirely even when toxicity spikes.
- **Capital and risk limits**: hard stops force unwinds at inopportune times.
- **Model risk**: every model is wrong; the question is whether it's wrong in a way you've bounded.
- **Liquidity asymmetries**: deep ATM, thin wings; the hedging basis is tenor- and strike-specific.
- **Pin risk and gamma squeezes**: concentrated open interest near expiry creates discontinuous P&L.
- **Regime transitions**: a book calibrated to low-vol regime pays on the transition to high-vol.

No single difficulty is insurmountable. The difficulty is that all of them are simultaneous and coupled.

---

## 10. Approaches to solving OMM

### 10.1 Classical stochastic control

A-S and its extensions (Cartea-Jaimungal, Guéant). Produces interpretable policies with clean decompositions (reservation price + optimal spread). Limitations: requires tractable model assumptions (diffusion, exponential fill intensity, quadratic risk penalty); scales badly to the full options surface; doesn't handle POMDP structure natively.

### 10.2 Reinforcement learning

DQN, PPO, SAC, actor-critic — trained in simulation against realistic order-flow models. Pros: handles POMDP and high-dimensional action space; can learn from realistic microstructure data. Cons: simulation-to-reality gap is large; exploration in production is expensive; interpretability is poor. State of the art (2024–2026): Coache-Cartea-Jaimungal conditionally elicitable risk measures; Jerome-Palmer-Savani learned beta policies (ICAIF 2024); production systems at several firms but gains modest vs. well-tuned classical policies.

### 10.3 Deep hedging (Buehler et al. 2019)

Neural networks trained to hedge in incomplete markets. Input: state (spot, vol, Greeks, maybe order book). Output: hedge action. Loss: convex risk measure (e.g., CVaR) of terminal P&L. Handles transaction costs, discrete hedging, multiple hedge instruments, and market incompleteness by construction. Deep Bellman Hedging (Buehler-Murray-Wood 2024) extends to actor-critic with options as hedge instruments. Practical adoption: growing at top firms for exotics and structured products; vanilla flow desks still use classical hedging with ML-enhanced signals.

### 10.4 POMDP-aware and multi-agent methods

Belief-state policies: maintain a posterior over hidden state (informed/uninformed, regime) and condition actions on belief. Multi-agent: treat other MMs and informed traders as strategic players; use mean-field or multi-agent RL. Academic frontier (Cartea-Jaimungal-Sánchez-Betancourt 2024 Nash equilibrium papers); limited production deployment.

### 10.5 The role of simulation

Realistic simulation is the hardest engineering problem in OMM research. Components:
- **Hawkes processes** for self-exciting trade arrivals.
- **Order book reconstruction** from OPRA feed.
- **Market impact models**: permanent and temporary impact calibrated to historical fills.
- **Regime-switching models**: HMMs over vol regimes for training data diversity.
- **Adversarial augmentation**: train RL policies against informed-trader adversaries.

Walk-forward backtesting with path-dependency (your fills alter the order book) is essential. Any simulator that uses historical fills as if they were independent of the MM's actions overestimates performance.

### 10.6 The publicly-known firm landscape (2024-2026)

| Firm | Public positioning |
|---|---|
| **Citadel Securities** | Self-reports ~30% of U.S. listed options; largest retail options wholesaler; options ADV +100% 2020→2025; 2024 net trading revenue $9.7B. Primarily internalized retail flow + lit MM. |
| **Jane Street** | 2024 trading revenue **$20.5B** (2× 2023); ~8% of OCC options volumes. Dominant in ETF arbitrage and index derivatives. SEBI July 2025 interim order in India (contested, resolved July 21 2025) prompted broader industry debate on expiry-day strategies. |
| **Susquehanna (SIG)** | Private; historically dominant in single-stock options on pro-rata venues; deep bench of option theorists. |
| **Optiver** | 2024 trading revenue $3.8B; expanding NYC presence 2025 to compete directly with Citadel/Jane Street/SIG on U.S. home turf. |
| **IMC** | Amsterdam-based, global; strong in ETF and index options. |
| **Virtu (public, VIRT)** | FY2025 revenues $3.63B (+26%); net income $912M; EPS $5.13. Publicly supported IEX Options proposal. |
| **Jump, Wolverine, Akuna, Belvedere, Flow Traders, Old Mission, Group One, Peak6, Two Sigma Securities** | Active mid-tier; specialists in various niches. |

Public disclosures are scarce beyond Virtu (public company) and press leaks; treat all numbers as best estimates.

---

## 11. Key tensions and insights

**Spread width.** The A-S formula gives the theoretically optimal spread, but in practice spread width is set by competition (you can only be as wide as the best competing quote if you want fills). The real optimization is: given competitor-imposed maximum spread, how do you allocate risk aversion $\gamma$ to avoid accumulating toxic inventory?

**Speed vs. thoughtfulness.** Under a latency budget of microseconds, complex models (rough vol, jump calibration) cannot run per-tick. Architecture: pre-compute a rich surface offline every few seconds; use lightweight delta-adjustments on every tick. The offline model sets the fair value; the online logic only adjusts for spot moves and inventory.

**Inventory carry vs. opportunity.** Hedging locks in edge but costs spread; not hedging preserves optionality on mean reversion but risks a disaster. The dominant modern practice is **statistical hedging**: hedge aggregate book Greeks at risk-based intervals, not per-contract.

**Spread capture vs. VRP vs. flow economics.** The honest decomposition of OMM P&L:
- 40-60% from spread capture on uninformed flow.
- 20-40% from implicit VRP (you accumulate short vol over time through flow imbalance and the VRP pays).
- 10-30% from rebates, PFOF, and tiered-fee optimization.
- Adverse-selection losses run 10-20% of gross, so net is the above minus this.

These fractions vary by firm and regime. In low-vol normal regimes VRP is stable and spread capture dominates. In crises VRP P&L is massively negative (short vol blows up) and only firms with robust risk limits and hedge capacity survive.

**Mean reversion vs. trend in different axes.** Underlying prices are close to martingales intraday (no persistent alpha). Vol is mean-reverting at longer horizons but trending at shorter ones (vol clustering). Order flow is self-exciting (Hawkes). These different timescales matter: an MM who treats all signals as mean-reverting loses when trending, and vice versa.

---

## 12. What experts actually obsess over

Stripping away the textbook derivations, the real edge in OMM — where top-tier firms separate from mid-tier — concentrates in a small number of places.

**Fair-value fidelity in the first 10 milliseconds after a tape event.** Most of the P&L difference between a good MM and a great MM is what happens in the 1-10 ms after an ES futures tick, an SPY print, or a correlated name moves. The first firm to re-price the SPX surface correctly captures spread; the second gets picked off. This is an infrastructure game (FPGA pricing, co-location, direct feeds) more than a model game.

**Bucketed vega management at book level, not contract level.** The difference between "I'm vega-flat" and "I'm vega-flat by tenor and skew bucket under a reasonable shock family" is the difference between surviving Aug 5 2024 and closing the fund. Every serious OMM obsesses over the vega-bucket risk matrix and its principal components.

**Flow segmentation.** Knowing which counterparties are informed, which are noise, and structuring your quoting to differentiate between them, is worth more than any model upgrade. This is done through venue routing, minimum-quantity filters, tier-based pricing, and real-time toxicity scoring of counterparty IDs where disclosed. The Citadel/Virtu retail PFOF stack exists because segmentation is so valuable.

**Adverse-selection cost as a first-class line item.** Top desks compute per-trade adverse-selection cost (post-trade mark-to-market over the next $N$ seconds) and feed it back into quoting. Quotes that have historically been adversely selected widen automatically. This is the Glosten-Milgrom update made operational.

**The hedging execution stack.** Once you know what to hedge, *how* you hedge — smart routing of ES/SPY/options hedges across venues, participation-rate control, impact-aware scheduling — separates firms whose hedging is cost-neutral from firms whose hedging is a P&L drain.

**Regime detection.** Knowing you're in a regime transition *before* it's obvious (low-vol to high-vol, trending to mean-reverting flow, normal to crisis correlation) lets you tighten risk in time. This is part HMM/ML, part macro judgment, part institutional memory. The firms that survive vol events without being forced to liquidate are the ones with regime-aware risk limits that contract pre-emptively.

**The obsession with path-dependency in backtesting.** Every serious quant knows that a backtest where your fills don't affect the book is worthless. The test that matters is: simulate your policy against an order-flow model that responds to your quotes, over thousands of paths, with realistic impact. The bar is higher than most academic papers meet.

**Capital efficiency.** The leading firms run much tighter capital per unit risk than competitors, which lets them size up relative to their vega limits. This is partly cross-asset netting (prime broker haircuts on a book with SPX, ES, SPY, VIX as one portfolio), partly risk-model sophistication, partly regulatory arbitrage across jurisdictions. The edge isn't glamorous but it's durable.

**Persistent humility about the tails.** Every year someone discovers a new tail. 1987, LTCM 1998, 2008, Flash Crash 2010, Volmageddon 2018, COVID 2020, GameStop 2021, rate shock 2022, SVB 2023, Yen-mageddon Aug 2024, Liberation Day April 2025. The firms that survive *expect* to be wrong about the next one and size accordingly. The firms that blow up had a view that their model was right.

---

## Further reading

**Books.** Gatheral, *The Volatility Surface* (2006) — the canonical surface treatment. Cartea, Jaimungal, Penalva, *Algorithmic and High-Frequency Trading* (2015) — the standard stochastic-control reference for market making. Sinclair, *Volatility Trading* (2nd ed. 2013) and *Option Trading* (2010). Hasbrouck, *Empirical Market Microstructure* (2007). Bayer, Friz, Fukasawa, Gatheral, Jacquier, Rosenbaum (eds.), *Rough Volatility* (SIAM, 2023/2024) — the first comprehensive book on the topic. Hull, *Options, Futures, and Other Derivatives* for breadth.

**Foundational papers.** Glosten-Milgrom (1985) *JFE* on adverse-selection spreads. Kyle (1985) *Econometrica* on informed trading and lambda. Ho-Stoll (1981) *JFE* inventory model. Avellaneda-Stoikov (2008) *Quantitative Finance* on HJB quoting. Dupire (1994) *Risk* local vol. Heston (1993) *RFS* stoch vol. Merton (1976) *JFE* jump diffusion. Bates (1996) on stoch vol + jumps. Carr-Madan (1999) FFT option pricing and (2001) static hedging. Gatheral (2004) SVI; Gatheral-Jacquier (2014) SSVI arbitrage-free surfaces. Hagan et al. (2002) SABR. Leland (1985) on transaction costs in hedging. Whalley-Wilmott (1997) asymptotic analysis of transaction-cost hedging. Easley-Kiefer-O'Hara (1996) PIN; Easley-López de Prado-O'Hara (2012) VPIN. Gatheral-Jaisson-Rosenbaum (2018) "Volatility is rough." Buehler-Gonon-Teichmann-Wood (2019) "Deep Hedging" *QF*.

**Recent (2023–2026).** Buehler-Murray-Wood, "Deep Bellman Hedging" (arXiv 2207.00932, v4 2024). Gatheral, "10 Years of Rough Volatility" (Bologna, April 2024). Dim-Eraker-Vilkov, "0DTEs: Trading, Gamma Risk and Volatility Propagation" (SSRN 2023, rev. 2024). Almeida-Freire-Hizmeri, "0DTE Asset Pricing" (Princeton 2024). Coache-Cartea-Jaimungal, "Conditionally Elicitable Dynamic Risk Measures for Deep RL" *SIAM JFM* 14(4), 2023. Cartea-Jaimungal-Sánchez-Betancourt, "Nash Equilibrium between Brokers and Traders" (arXiv 2407.10561). Doshi-Patel-Stephens, "Risky Intraday Order Flow and Option Liquidity" (May 2025). Cboe white paper, "0DTE Index Options and Market Volatility" (2024). The *Deep Learning in Finance* survey (IJTAF 2025) for deep hedging / deep calibration overview.

**Practitioner resources.** Cboe Product Specifications for SPX/SPXW/XSP; OCC options rules; CBOE's "Cboe Insights" blog; SEC rule filings (especially on auction mechanisms, order types, and the 2025 IEX Options proceedings); OPRA feed specs; Risk.net for industry perspective; the annual FIA Boca conferences for regulatory and exchange-structure updates.

---

## Conclusion

OMM is a problem where the **mathematics is elegant** (HJB, Bayesian updating, surface arbitrage-freeness), the **statistics are brutal** (non-Gaussian, non-stationary, regime-shifting), and the **engineering determines survival** (latency, risk systems, capital efficiency). The Avellaneda-Stoikov decomposition — inventory shifts the reservation price, microstructure sets the spread — is the correct mental anchor, but a real book lives on a high-dimensional Greek manifold where vega bucketing, cross-Greeks, and adverse-selection detection matter more than the scalar delta-hedging problem most treatments focus on.

The structural edges are **three stacked P&L streams**: spread capture on uninformed flow (segmentable via venue routing and PFOF economics), variance risk premium harvesting (implicit, regime-dependent, catastrophic in tails), and rebate/fee economics. The adversarial cost is adverse selection, managed by Bayesian updating, toxicity detection, and asymmetric quote fading — never by trying to avoid informed flow entirely.

In 2025-2026, the dominant structural change is **0DTE**: 59% of SPX volume is same-day expiry, reshaping intraday vol dynamics and forcing every SPX MM to rebuild its gamma-management and end-of-day hedging stack. The exchange landscape continues to fragment (18 venues, IEX Options and MEMX MX2 pending), regulatory winds have shifted with the Atkins SEC withdrawing the Order Competition Rule, and consolidation among top MMs (Citadel, Jane Street, SIG, Optiver, Virtu) continues as the infrastructure arms race raises the minimum viable scale.

The honest final observation: a senior quant researcher entering this field in 2026 should expect that **the equation is not the edge**. The published papers give you the common knowledge. The edge lives in the integration — in how fast your fair value updates on a correlated tick, how faithfully your vega buckets are managed in stress, how cleanly you segment flow by toxicity, and how well your risk limits contract *before* the regime shifts rather than after. The firms that dominate are the ones where the mathematics, the data, the engineering, and the institutional memory all work together, and where the whole system has been battle-tested across enough tail events to know which models to distrust and which risk limits to actually believe.