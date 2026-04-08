## Simulation

The simulaiton environment is the fondation of this entire project. This will produce all of the information that the Agent will use to make decisions. Therefore, it is critical that the simulation is a good approximation for ture financial markets if the Agent's actions are to be viewed in good light.

The foundation of this simulation is the spot price path. We want to create a spot asset price path that is realistic becuase this is the main tool that the Agent will use to directly hedge delta exposure. Ideally, this simulation also shares all or some of the 11 stylized facts that Cont outlines in his paper *Empirical properties of asset returns:*
*stylized facts and statistical issues*. 

There also needs to be a way to represnt transaction costs on the spot asset. The way that I will implement this to start is by a percentage of the notional value of the trade. This percetage will be represented by $k$. So the transation cost of a trade is $k\cdot S\cdot n_{shares}$.

This will also be the foundation of how we frame time for the entire project. Time will be measured in time steps. Each time step will represent a trading day. I am choosing this framework because it is simple and straight forward. The Black-Scholes expects time in years so I can just multiply (total_steps - current_time_step) * 1/252 to easiyl get $\tau$.  I am planning on simulating one month of trading with options that expire at the end of the month. These means that the sim will only be 252/12=21 steps which will make computation very quick. If I want to decrease the unit of time of each step in the future I think it would make for an interesting sensitivity analysis.

Once these this are implemented then we should be able to writed a function that outputs the spot price at each time step, then we should be able to run this for some timesteps and plot the price of the spot. This is represented by th following equation:
$$
S_{t+1}=S_{t}\cdot e^{(r-\frac{1}{2}\sigma^2)dt + \sigma \sqrt{dt}Z_t}, Z\sim N(1,0)
$$
In order to create non-constant volatility, $\sigma$ will be determined by a hidden Markov chain.

### Regime Switching

**Look at end of chat of "Final project proposal for Sunberg"**

The stationary distribution π satisfies π = πT. For a 3-state no-skip chain (low↔high = 0), this gives us a system we can solve analytically. Given target π = [0.60, 0.33, 0.07], let's call the off-diagonal entries:

- `a` = P(low → medium)
- `b` = P(medium → low)
- `c` = P(medium → high)
- `d` = P(high → medium)

The stationary equations are:

- π₁ = π₁(1-a) + π₂·b → π₁·a = π₂·b
- π₃ = π₂·c + π₃(1-d) → π₃·d = π₂·c
- π₂ = π₁·a + π₂(1-b-c) + π₃·d (automatically satisfied)

From equation 1: `b = π₁·a / π₂ = (0.60/0.33)·a ≈ 1.818·a`

From equation 2: `c = π₃·d / π₂ = (0.07/0.33)·d ≈ 0.212·d`

So we have two free parameters — `a` (controls how quickly calm breaks) and `d` (controls how quickly crisis resolves). We need to choose them to get realistic regime durations. Mean duration of a regime is `1/(exit rate)`:

- Mean days in low = 1/a → target ~43 days (from Cerboni Baiardi) → a ≈ 0.023 ✓ (already correct)
- Mean days in high = 1/d → target ~37 days → d ≈ 0.027 ✓ (already correct)

So those are fine. The problem is `c`. With `a = 0.023` and `d = 0.027`:

- `b = 1.818 × 0.023 ≈ 0.042`
- `c = 0.212 × 0.027 ≈ 0.006`

The math is internally consistent — `c = 0.006` is actually *correct* given those constraints. The real issue is that 7% of 1000 days is only 70 expected high-vol days, and with high regime stickiness it clusters — so in any single 1000-day run you can get unlucky. Your 0.2% result (2 days) is just bad luck from a single seed.

Let's verify this by running many paths and checking the average:

Run regime distribution check across many episodes

Julia isn't available in this environment, so I can work through the math directly. Let me compute what the stationary distribution actually is for the current matrix:

For the current matrix with `a=0.023, b=0.027, c=0.006, d=0.027`, solving πT = π:

- π₁·0.023 = π₂·0.027 → π₁/π₂ = 0.027/0.023 ≈ 1.174
- π₂·0.006 = π₃·0.027 → π₂/π₃ = 0.027/0.006 = 4.5

Normalizing: π₂ = π₁/1.174, π₃ = π₂/4.5 = π₁/5.283

π₁ + π₁/1.174 + π₁/5.283 = 1 → π₁(1 + 0.852 + 0.189) = 1 → π₁ ≈ **0.493**, π₂ ≈ **0.420**, π₃ ≈ **0.093**

So the actual stationary distribution of the current matrix is 49/42/9, **not** 60/33/7. The matrix from the paper doesn't reproduce the paper's own stated stationary distribution at our parameter values — which explains the weird empirical results.

To hit exactly 60/33/7, we need to solve for the off-diagonals properly. With target π = [0.60, 0.33, 0.07] and keeping mean regime durations (1/a ≈ 43 days, 1/d ≈ 37 days → a=0.023, d=0.027):

- `b = (π₁/π₂)·a = (0.60/0.33)·0.023 ≈ **0.042**`
- `c = (π₃/π₂)·d = (0.07/0.33)·0.027 ≈ **0.006**` (unchanged)

So `b` needs to go from 0.027 to 0.042 — medium returns to low faster. Let's update the matrix:



## Black Scholes and OptionPosition Struct

The Agent will need to be able to have a portfolio of options contracts. This will be represented by a struct that hold the strike price, the original duration of the option, the quantity of this type of contract and whether or not these are calls or puts. This project focuses on just a single strike and a single expiry so the entire portfolio will be a vector or tuple of two OptionPosition structs (one for the calls and one for the puts).

At each time-step the Agent will need to calculate the prices and greeks for each of its option positions.



| Variables |                       |
| --------- | --------------------- |
| $\tau$    | Time until expiration |
|           |                       |
|           |                       |

