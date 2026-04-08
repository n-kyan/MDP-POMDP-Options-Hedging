# Implementation Plan: Optimal Options Hedging Under Uncertainty

## How to Use This Document

This plan is structured top-down. **Layer 1** is the architecture overview — read this to understand how everything fits together. **Layer 2** breaks each module into structs, function signatures, and data flow. **Layer 3** gives pseudocode, Julia idioms, and testing strategies for the hard parts. Each section references the relevant chapter in **Kochenderfer et al., "Algorithms for Decision Making"** (abbreviated **ADM**) so you can read the theory alongside the implementation.

---

# LAYER 1: Architecture Overview

## Module Map


Ok i am on board with the transition probabilities and the length of the single episode. One question i still have is if all of the episodes start in a calm regime. I think this could skew the results pretty significantly because i think it could learn that it is certainly in a calm regime at the begining. Can you explain this to me and make sure that we have a solution for this.



After that I it seems that the spot market sim is complete. I think that the next step of the simulation is the inventory of the agent and the simulated lifts of the bids and asks



The project has **7 modules** organized into 3 tiers:

```
TIER 1 — FOUNDATION (no dependencies on other project modules)
┌─────────────────────────────────────────────────────────────┐
│  Module 1: BlackScholes                                     │
│  Option pricing, Greeks (delta, gamma, vega), BS formulas   │
│                                                             │
│  Module 2: MarketSim                                        │
│  GBM price paths, regime-switching dynamics, Markov chain   │
└─────────────────────────────────────────────────────────────┘

TIER 2 — ENVIRONMENT (depends on Tier 1)
┌─────────────────────────────────────────────────────────────┐
│  Module 3: HedgingEnv                                       │
│  MDP/POMDP environment definition.                          │
│  State, actions, transitions, reward function.              │
│  POMDPs.jl generative interface.                            │
│  Includes state discretization for value iteration.         │
└─────────────────────────────────────────────────────────────┘

TIER 3 — SOLVERS & EVALUATION (depends on Tiers 1-2)
┌─────────────────────────────────────────────────────────────┐
│  Module 4: Solvers                                          │
│  Value iteration (from scratch)                             │
│  DQN (from scratch, Flux.jl for NN)                         │
│  MCTS (POMDPs.jl / MCTS.jl)                                │
│  QMDP (from scratch, uses value iteration output)           │
│  Online POMDP (POMCPOW from POMDPs.jl)                     │
│                                                             │
│  Module 5: BeliefUpdaters                                   │
│  Exact Bayesian discrete filter                             │
│  Particle filter                                            │
│                                                             │
│  Module 6: Baselines                                        │
│  Black-Scholes delta hedge policy                           │
│                                                             │
│  Module 7: Evaluation                                       │
│  Run simulations, collect metrics, plot distributions       │
└─────────────────────────────────────────────────────────────┘
```

## Dependency Graph (Build Order)

```
BlackScholes ──┐
               ├──> HedgingEnv ──┬──> Solvers ──────┐
MarketSim ─────┘                 ├──> Baselines      ├──> Evaluation
                                 ├──> BeliefUpdaters─┘
                                 └───────────────────────> Evaluation
```

**Critical path:** You cannot test anything interesting until `HedgingEnv` works. And `HedgingEnv` cannot work until `BlackScholes` and `MarketSim` are correct. So the foundation must be rock-solid before you move on.

## File Structure

```
OptionsHedging/
├── Project.toml
├── Manifest.toml
├── src/
│   ├── OptionsHedging.jl          # Main module, exports everything
│   ├── black_scholes.jl           # Module 1
│   ├── market_sim.jl              # Module 2
│   ├── hedging_env.jl             # Module 3
│   ├── discretization.jl          # State discretization (part of Module 3)
│   ├── solvers/
│   │   ├── value_iteration.jl     # Level 1
│   │   ├── mcts_interface.jl      # Level 2 (thin wrapper for MCTS.jl)
│   │   ├── dqn.jl                 # Level 2
│   │   ├── qmdp.jl               # Level 3
│   │   └── pomdp_interface.jl     # Level 3 (wrapper for POMCPOW)
│   ├── belief_updaters.jl         # Module 5 (Level 3)
│   ├── baselines.jl               # Module 6
│   └── evaluation.jl              # Module 7
├── scripts/
│   ├── run_level1.jl              # Run Level 1 experiments
│   ├── run_level2.jl              # Run Level 2 experiments
│   ├── run_level3.jl              # Run Level 3 experiments
│   └── generate_figures.jl        # Paper figures
└── test/
    ├── test_black_scholes.jl
    ├── test_market_sim.jl
    ├── test_hedging_env.jl
    └── test_solvers.jl
```

## Package Dependencies

```toml
[deps]
POMDPs = "a]93ec93-..."        # Core interface
POMDPTools = "..."              # Simulators, policies, utilities
MCTS = "..."                    # Monte Carlo Tree Search (Level 2)
POMCPOW = "..."                 # Online POMDP solver (Level 3)
QMDP = "..."                   # QMDP solver (Level 3, or implement from scratch)
Distributions = "..."           # Normal, Categorical, etc.
Flux = "..."                    # Neural networks for DQN (Level 2)
Plots = "..."                   # Visualization
StatsBase = "..."               # Histogram, quantile, etc.
Random = "..."                  # stdlib — RNG control
LinearAlgebra = "..."           # stdlib — dot products, norms
```

## Weekly Schedule

| Week | Dates | Goal | Modules |
|------|-------|------|---------|
| 1 | Mar 30 – Apr 6 | Foundation + Level 1 solver | BlackScholes, MarketSim, HedgingEnv, ValueIteration, Baselines, Evaluation (basic) |
| 2 | Apr 7 – Apr 13 | Level 1 complete, start Level 2 | Level 1 experiments, paper figures. Begin MCTS integration. |
| 3 | Apr 14 – Apr 20 | Level 2 complete | DQN training, MCTS tuning, Level 2 experiments and comparisons |
| 4 | Apr 21 – Apr 27 | Level 3 attempt + paper | BeliefUpdaters, QMDP, POMCPOW. Paper writing. |
| 5 | Apr 28 – Apr 30 | Polish and submit | Final paper edits, figure cleanup, code cleanup |

---

# LAYER 2: Module-by-Module Breakdown

---

## Module 1: BlackScholes (`black_scholes.jl`)

**Purpose:** Compute option prices and Greeks analytically. These are pure math functions with zero dependencies.

**Textbook reference:** This is finance, not in ADM. Standard Black-Scholes formulas. If you need a refresher, Hull's "Options, Futures, and Other Derivatives" Ch. 15 has everything, or any online BS formula reference.

### Structs

```julia
struct OptionContract
    K::Float64          # Strike price
    T::Float64          # Time to expiry (in years)
    r::Float64          # Risk-free rate (annualized)
    q::Float64          # Dividend yield (0.0 for simplicity)
    is_call::Bool       # true = call, false = put
end
```

No mutable state here. An `OptionContract` is a fixed description of the option being hedged.

### Functions

```julia
# Core pricing
bs_price(S, σ, contract, τ) -> Float64
# S = spot price, σ = volatility, τ = time remaining to expiry
# Returns the Black-Scholes theoretical value

# Greeks
bs_delta(S, σ, contract, τ) -> Float64
# Sensitivity of price to spot. Range: [0,1] for calls, [-1,0] for puts.

bs_gamma(S, σ, contract, τ) -> Float64
# Rate of change of delta. Always positive. Peaks near ATM, near expiry.

bs_vega(S, σ, contract, τ) -> Float64
# Sensitivity to volatility. Useful for analysis, not required for hedging.

# Helper (internal)
_d1(S, K, r, q, σ, τ) -> Float64
_d2(S, K, r, q, σ, τ) -> Float64
```

### Data Flow

- **Inputs:** Spot price `S`, volatility `σ`, and an `OptionContract` + remaining time `τ`
- **Outputs:** Scalar values (price, delta, gamma)
- **Called by:** `HedgingEnv` (to compute Greeks at each timestep), `Baselines` (to compute BS delta hedge target)

---

## Module 2: MarketSim (`market_sim.jl`)

**Purpose:** Generate price paths for the underlying asset. Two modes: constant-vol GBM (Level 1) and regime-switching (Level 2+).

**Textbook reference:** ADM Ch. 19.1 (belief initialization, Markov chains for regime transitions). The GBM itself is standard stochastic processes.

### Structs

```julia
struct GBMParams
    μ::Float64          # Annualized drift
    σ::Float64          # Annualized volatility (constant)
    dt::Float64         # Timestep size (in years), e.g. 1/252 for daily
end

struct RegimeSwitchingParams
    μ::Float64                      # Drift (same across regimes for simplicity)
    σ_levels::Vector{Float64}       # Volatility for each regime, e.g. [0.12, 0.35]
    transition_matrix::Matrix{Float64}  # Row-stochastic Markov transition matrix
    dt::Float64                     # Timestep size
end
```

### Functions

```julia
# Single-step price update under GBM
step_gbm(S, params::GBMParams, rng) -> Float64
# Returns S_{t+1} given S_t

# Single-step price update under regime-switching
step_regime(S, regime, params::RegimeSwitchingParams, rng) -> (S_new, regime_new)
# Returns new price AND new regime (regime may or may not have switched)

# Full path generation (for testing and evaluation, NOT used by the MDP)
simulate_gbm_path(S0, params::GBMParams, n_steps, rng) -> Vector{Float64}
simulate_regime_path(S0, regime0, params::RegimeSwitchingParams, n_steps, rng) -> (prices, regimes)
```

### Data Flow

- **Inputs:** Current price, current regime (if applicable), params, RNG
- **Outputs:** New price (and new regime)
- **Called by:** `HedgingEnv` (inside `gen()` for single-step transitions), `Evaluation` (full path generation for batch testing)

### Key Implementation Detail

The single-step GBM update is:

```
S_{t+1} = S_t * exp((μ - σ²/2) * dt + σ * √dt * Z)
```

where Z ~ Normal(0,1). Use the log-return formulation (geometric, not arithmetic) to prevent negative prices.

For regime switching, first sample the new regime from the transition matrix, then use that regime's σ to generate the price step.

---

## Module 3: HedgingEnv (`hedging_env.jl` + `discretization.jl`)

**Purpose:** This is the core of the project. It defines the MDP (and later POMDP) that wraps the simulation environment. It implements the POMDPs.jl generative interface so that MCTS, POMCPOW, and POMDPs.jl simulators can interact with it.

**Textbook reference:** ADM Ch. 7.1 (MDP definition), Ch. 9.6 (generative interface for MCTS), Ch. 20.1 (belief-state MDP for POMDP).

### Structs

```julia
# === MDP State ===
struct HedgingState
    δ_net::Float64       # Net portfolio delta (option delta - hedge shares)
    Γ::Float64           # Portfolio gamma
    m::Float64           # Moneyness S/K
    τ::Float64           # Time to expiry remaining
    regime::Int           # Volatility regime index (1, 2, ...). For Level 1, always 1.
    S::Float64           # Raw spot price (needed for txn cost calculation, NOT part of discretized state)
    hedge_shares::Float64 # Shares of underlying currently held as hedge
end

# === MDP Definition (Level 1: constant vol) ===
struct HedgingMDP <: MDP{HedgingState, Int}
    contract::OptionContract
    sim_params::Union{GBMParams, RegimeSwitchingParams}
    κ::Float64           # Transaction cost parameter
    λ::Float64           # Risk penalty parameter
    γ_discount::Float64  # Discount factor
    n_options::Int        # Number of option contracts held (e.g., 1)
    actions::Vector{Float64}  # Hedge fractions [0.0, 0.25, 0.5, 0.75, 1.0]
end

# === POMDP Definition (Level 3: hidden regime) ===
struct HedgingPOMDP <: POMDP{HedgingState, Int, Float64}
    # Same fields as MDP, plus observation model
    contract::OptionContract
    sim_params::RegimeSwitchingParams
    κ::Float64
    λ::Float64
    γ_discount::Float64
    n_options::Int
    actions::Vector{Float64}
end
```

**Design decision — why `Int` for actions:** The action type is `Int` (an index into the `actions` vector) because POMDPs.jl solvers and MCTS work best with discrete, enumerable action spaces. The mapping from action index to hedge fraction is: `hedge_frac = mdp.actions[a]`.

**Design decision — why `S` and `hedge_shares` are in the state:** Although the *discretized* MDP state for value iteration only uses (δ_net, Γ, m, τ, regime), the full `HedgingState` needs `S` and `hedge_shares` for computing the exact transaction cost and P&L. The generative interface (`gen`) works with the full continuous state. For value iteration, you discretize/project onto the grid.

### POMDPs.jl Interface Functions to Implement

For the **MDP** (Levels 1-2), you need:

```julia
POMDPs.actions(mdp::HedgingMDP) -> Vector{Int}
# Returns [1, 2, 3, 4, 5] (indices into mdp.actions)

POMDPs.discount(mdp::HedgingMDP) -> Float64
# Returns mdp.γ_discount

POMDPs.isterminal(mdp::HedgingMDP, s::HedgingState) -> Bool
# Returns true when s.τ <= 0

POMDPs.initialstate(mdp::HedgingMDP) -> distribution
# Returns a distribution over initial states.
# Deterministic: start at S=K (ATM), hedge_shares=0, regime=1

POMDPs.gen(mdp::HedgingMDP, s::HedgingState, a::Int, rng::AbstractRNG) -> (sp=..., r=...)
# THE CRITICAL FUNCTION. This is the one-step simulator.
# See pseudocode in Layer 3.
```

For the **POMDP** (Level 3), you additionally need:

```julia
POMDPs.gen(pomdp::HedgingPOMDP, s::HedgingState, a::Int, rng::AbstractRNG) -> (sp=..., o=..., r=...)
# Same as MDP gen, but also returns an observation.
# The observation is the log-return: o = log(S_new / S_old)
# The agent cannot see s.regime in the POMDP.
```

### Discretization (`discretization.jl`)

For value iteration (Level 1), you need to discretize the continuous state into a grid.

```julia
struct StateGrid
    δ_bins::Vector{Float64}     # Bin centers for net delta
    Γ_bins::Vector{Float64}     # Bin centers for gamma
    m_bins::Vector{Float64}     # Bin centers for moneyness
    τ_bins::Vector{Float64}     # Bin centers for time to expiry
    regime_bins::Vector{Int}    # [1] for Level 1, [1,2] for Level 2
end

# Convert continuous state to grid index
function state_to_index(s::HedgingState, grid::StateGrid) -> Int
    # Find nearest bin for each dimension, return linear index
end

# Convert grid index back to representative state
function index_to_state(idx::Int, grid::StateGrid) -> HedgingState
    # Reverse mapping
end

# Total number of discrete states
n_states(grid::StateGrid) -> Int
```

**Choosing grid resolution:** Start coarse (maybe 8-10 bins per dimension) to get things working, then refine. With 4 state dimensions at 10 bins each and 2 regimes, you have 10×10×10×10×2 = 20,000 states. With 5 actions, value iteration needs to iterate over 100,000 state-action pairs per sweep — fast enough in Julia.

### Suggested Bin Ranges

| Dimension | Range | Reasoning |
|-----------|-------|-----------|
| δ_net | [-1.0, 1.0] | Delta of a single call is [0,1], so net delta after partial hedging is bounded |
| Γ | [0, 0.05] | Gamma is typically small in absolute terms. Compute empirically from BS formula to calibrate. |
| m (S/K) | [0.8, 1.2] | Covers ITM to OTM. Most hedging action happens near ATM. |
| τ | [0, T] | T is the option's total life, e.g. 30 days = 30/252 ≈ 0.119 years |

**Important:** These ranges are starting guesses. Run the GBM simulation 1000 times and look at the empirical distribution of each state variable to calibrate the bins. This is one of your first tasks.

---

## Module 4: Solvers (`solvers/`)

### 4a: Value Iteration (`value_iteration.jl`) — Level 1

**Textbook reference:** ADM Ch. 7.5 (Value Iteration), Algorithm 7.6.

```julia
struct ValueIterationSolver
    grid::StateGrid
    max_iterations::Int     # e.g. 100
    tolerance::Float64      # e.g. 1e-4 for Bellman residual convergence
end

struct ValueIterationPolicy
    V::Array{Float64}       # Value function over discretized states
    Q::Array{Float64}       # Q-values: states × actions (keep for QMDP later)
    policy::Array{Int}      # Optimal action index at each state
    grid::StateGrid         # The grid used (needed to map continuous states to indices)
end

function solve_vi(mdp::HedgingMDP, solver::ValueIterationSolver, rng) -> ValueIterationPolicy
    # See Layer 3 for pseudocode
end

# Use the policy: map a continuous state to the grid, look up the action
function get_action(policy::ValueIterationPolicy, s::HedgingState) -> Int
    idx = state_to_index(s, policy.grid)
    return policy.policy[idx]
end
```

### 4b: MCTS Interface (`mcts_interface.jl`) — Level 2

**Textbook reference:** ADM Ch. 9.6 (Monte Carlo Tree Search).

This is a thin wrapper. Because your `HedgingMDP` already implements the POMDPs.jl generative interface, MCTS.jl can use it directly.

```julia
using MCTS

function create_mcts_planner(mdp::HedgingMDP;
    n_iterations=1000,
    depth=20,
    exploration_constant=5.0
) -> AbstractMCTSPlanner

    solver = DPWSolver(
        n_iterations=n_iterations,
        depth=depth,
        exploration_constant=exploration_constant,
        # DPW (Double Progressive Widening) handles continuous states
        # by limiting tree branching. Important tuning parameters:
        k_state=4.0,
        alpha_state=0.1,
    )
    return solve(solver, mdp)
end

# Usage in simulation loop:
# a = action(planner, s)
```

**Why DPWSolver instead of MCTSSolver:** Your state space is continuous (even though actions are discrete). Standard MCTS builds a tree node for each unique state, which doesn't work with continuous states (every state is unique). DPW (Double Progressive Widening) limits the number of child nodes using a progressive widening rule, making it work with continuous state spaces. This is the standard approach — ADM Section 9.6 discusses this.

### 4c: DQN (`dqn.jl`) — Level 2

**Textbook reference:** ADM Ch. 17.2 (Q-Learning), 17.6 (Action Value Function Approximation), 17.7 (Experience Replay). Also reference the original DQN paper (Mnih et al., 2015) for the target network idea.

```julia
using Flux

struct DQNParams
    hidden_sizes::Vector{Int}   # e.g. [64, 64]
    learning_rate::Float64      # e.g. 1e-3
    batch_size::Int             # e.g. 64
    buffer_size::Int            # Replay buffer capacity, e.g. 50_000
    γ::Float64                  # Discount factor
    ε_start::Float64            # Initial exploration rate, e.g. 1.0
    ε_end::Float64              # Final exploration rate, e.g. 0.05
    ε_decay_steps::Int          # Steps to decay ε over, e.g. 10_000
    target_update_freq::Int     # Steps between target network updates, e.g. 500
    n_episodes::Int             # Training episodes, e.g. 5_000
end

# Experience replay buffer
mutable struct ReplayBuffer
    states::Vector{Vector{Float64}}
    actions::Vector{Int}
    rewards::Vector{Float64}
    next_states::Vector{Vector{Float64}}
    dones::Vector{Bool}
    capacity::Int
    idx::Int                    # Current write position (circular buffer)
    size::Int                   # Number of stored experiences
end

mutable struct DQNAgent
    Q_online::Chain             # Flux neural network
    Q_target::Chain             # Target network (frozen copy, periodically updated)
    optimizer::Any              # Flux optimizer state
    buffer::ReplayBuffer
    params::DQNParams
    step_count::Int
end

function create_dqn_agent(state_dim::Int, n_actions::Int, params::DQNParams) -> DQNAgent
function select_action(agent::DQNAgent, state_vec::Vector{Float64}, rng) -> Int
    # ε-greedy: with probability ε pick random action, else pick argmax Q
end
function store_experience!(agent::DQNAgent, s, a, r, sp, done)
function train_step!(agent::DQNAgent, rng)
    # Sample batch from replay buffer
    # Compute target: r + γ * max_a' Q_target(sp, a')
    # Compute loss: MSE between Q_online(s,a) and target
    # Gradient step on Q_online
end
function update_target!(agent::DQNAgent)
    # Copy Q_online parameters to Q_target
end
function train_dqn!(agent::DQNAgent, mdp::HedgingMDP, rng) -> DQNAgent
    # Main training loop over episodes
end

# Convert HedgingState to a flat vector for the neural network
function state_to_vector(s::HedgingState) -> Vector{Float64}
    # [δ_net, Γ, m, τ, regime_onehot...]
    # Normalize each dimension to roughly [-1, 1] or [0, 1]
end
```

**Critical note on state normalization:** Neural networks train much better when inputs are normalized. Normalize each state dimension to approximately [0, 1] or [-1, 1] based on the expected ranges. Use the same ranges from your discretization bins. Without normalization, DQN training is likely to diverge or learn very slowly.

### 4d: QMDP (`qmdp.jl`) — Level 3

**Textbook reference:** ADM Ch. 21.1 (Fully Observable Value Approximation / QMDP).

QMDP is beautifully simple: it takes your Q-values from value iteration (one set per regime) and averages them using the belief.

```julia
struct QMDPPolicy
    Q_per_regime::Vector{Array{Float64}}  # Q-values from VI, one array per regime
    grid::StateGrid
end

function get_action_qmdp(policy::QMDPPolicy, s::HedgingState, belief::Vector{Float64}) -> Int
    # For each action a:
    #   Q_qmdp(b, a) = Σ_regime  belief[regime] * Q_per_regime[regime][state_idx, a]
    # Return argmax over a
end
```

To build this, you run value iteration **once per regime** (with that regime's volatility held fixed), collect the Q-tables, and then combine them using the belief at decision time. This is why keeping the `Q` array in `ValueIterationPolicy` matters — QMDP reuses it directly.

### 4e: Online POMDP (`pomdp_interface.jl`) — Level 3

**Textbook reference:** ADM Ch. 22.5-22.6 (Online POMDP methods, POMCPOW/DESPOT).

Like MCTS, this is a thin wrapper because your `HedgingPOMDP` implements the POMDPs.jl interface.

```julia
using POMCPOW

function create_pomcpow_planner(pomdp::HedgingPOMDP;
    tree_queries=1000,
    max_depth=20,
    criterion=MaxUCB(5.0)
)
    solver = POMCPOWSolver(
        tree_queries=tree_queries,
        max_depth=max_depth,
        criterion=criterion,
        # Additional tuning parameters
    )
    return solve(solver, pomdp)
end
```

---

## Module 5: BeliefUpdaters (`belief_updaters.jl`) — Level 3

**Textbook reference:** ADM Ch. 19.2 (Discrete State Filter), Ch. 19.6 (Particle Filter).

### Exact Bayesian Updater

```julia
mutable struct ExactBeliefUpdater
    belief::Vector{Float64}     # Probability of each regime, e.g. [0.6, 0.4]
    params::RegimeSwitchingParams
end

function update_belief!(updater::ExactBeliefUpdater, observed_return::Float64)
    # 1. Predict: b_pred[j] = Σ_i T[i,j] * b[i]
    # 2. Update: b_new[j] ∝ P(return | σ_j) * b_pred[j]
    #    where P(return | σ_j) = pdf(Normal(μ*dt, σ_j*√dt), observed_return)
    # 3. Normalize: b_new ./= sum(b_new)
end

function get_belief(updater::ExactBeliefUpdater) -> Vector{Float64}
function reset_belief!(updater::ExactBeliefUpdater, prior::Vector{Float64})
```

### Particle Filter

```julia
mutable struct ParticleFilterUpdater
    particles::Vector{Int}      # Each particle is a regime index
    weights::Vector{Float64}    # Importance weights
    n_particles::Int            # e.g. 1000
    params::RegimeSwitchingParams
end

function update_pf!(updater::ParticleFilterUpdater, observed_return::Float64, rng)
    # 1. Propagate: for each particle, sample new regime from transition matrix
    # 2. Weight: weight each particle by P(observed_return | σ of that particle's regime)
    # 3. Normalize weights
    # 4. Resample (systematic resampling) if effective sample size drops below threshold
end

function get_belief_pf(updater::ParticleFilterUpdater) -> Vector{Float64}
    # Convert particle distribution to regime probability vector
    # Count fraction of particles in each regime
end
```

---

## Module 6: Baselines (`baselines.jl`)

**Purpose:** The Black-Scholes delta hedge baseline.

```julia
struct BSHedgePolicy
    contract::OptionContract
    σ::Float64              # Volatility assumed by the BS hedger
    # Level 1: uses the true constant σ
    # Level 2-3: uses a fixed σ (e.g., long-run average)
    #            which is realistic — a BS hedger doesn't know the regime
end

function get_action_bs(policy::BSHedgePolicy, s::HedgingState) -> Int
    # Compute BS delta at current (S, τ, σ)
    # Target hedge position = n_options * bs_delta
    # Current hedge position = s.hedge_shares
    # "Ideal" action is to close the entire gap → action index 5 (100%)
    # BS always hedges 100% of delta exposure → return action index 5
end
```

**Note:** The BS baseline always takes action 5 (hedge 100%). That's the whole point — it doesn't reason about transaction costs. The comparison is: does the MDP/POMDP agent learn to do better by sometimes choosing to hedge less?

---

## Module 7: Evaluation (`evaluation.jl`)

**Purpose:** Run policies through simulated episodes, collect P&L paths, compute metrics, generate plots.

```julia
struct EpisodeResult
    pnl_series::Vector{Float64}    # Per-step net P&L
    cumulative_pnl::Float64        # Total P&L over the episode
    total_txn_cost::Float64        # Total transaction costs paid
    n_trades::Int                  # Number of non-zero hedge adjustments
    states::Vector{HedgingState}   # State trajectory (for debugging/visualization)
    actions::Vector{Int}           # Action trajectory
end

struct EvalMetrics
    mean_pnl::Float64
    std_pnl::Float64
    sharpe::Float64                # mean / std
    max_drawdown::Float64
    mean_txn_cost::Float64
    mean_n_trades::Float64
    pnl_5th_percentile::Float64    # Left tail
    pnl_95th_percentile::Float64   # Right tail
end

function run_episode(mdp, policy, rng) -> EpisodeResult
    # Initialize state from initialstate(mdp)
    # Loop until terminal:
    #   a = get_action(policy, s)   # or action(planner, s) for MCTS
    #   (sp, r) = gen(mdp, s, a, rng)
    #   record step
    #   s = sp
end

function evaluate_policy(mdp, policy, n_episodes, rng) -> (Vector{EpisodeResult}, EvalMetrics)
    # Run n_episodes, compute aggregate metrics
end

function compare_policies(mdp, policies::Dict{String, Any}, n_episodes, rng)
    # Run all policies on the SAME random seeds for fair comparison
    # Return side-by-side metrics
end

# Plotting
function plot_pnl_distributions(results::Dict{String, Vector{EpisodeResult}})
function plot_policy_heatmap(policy::ValueIterationPolicy, grid::StateGrid)
    # 2D heatmap: fix some state dims, vary 2, show action choice
    # E.g., x-axis = delta, y-axis = moneyness, color = action
function plot_txn_cost_sensitivity(results_by_kappa::Dict{Float64, EvalMetrics})
```

---

# LAYER 3: Algorithm Deep-Dives

---

## 3.1 The `gen` Function (The Heart of the Environment)

This is the single most important function in the project. Every solver calls it. If this is wrong, everything downstream is wrong.

**ADM Reference:** Ch. 7.1 (MDP transitions), Ch. 9.6 (generative interface for MCTS).

### Pseudocode for `gen(mdp::HedgingMDP, s::HedgingState, a::Int, rng)`

```
INPUT: current state s, action index a, random number generator rng
OUTPUT: NamedTuple (sp=new_state, r=reward)

1. DETERMINE HEDGE ACTION
   hedge_frac = mdp.actions[a]                  # e.g., 0.0, 0.25, ..., 1.0
   shares_to_trade = hedge_frac * s.δ_net * mdp.n_options  
       # This is the number of shares of underlying to buy/sell.
       # If δ_net > 0 (option is long delta), we sell shares to hedge.
       # If δ_net < 0, we buy shares.
       # The sign convention: shares_to_trade represents the CHANGE in hedge position.
   new_hedge_shares = s.hedge_shares + shares_to_trade

2. COMPUTE TRANSACTION COST
   txn_cost = mdp.κ * abs(shares_to_trade) * s.S

3. STEP THE PRICE FORWARD
   if mdp.sim_params isa GBMParams
       S_new = step_gbm(s.S, mdp.sim_params, rng)
       regime_new = s.regime  # unchanged
   else
       (S_new, regime_new) = step_regime(s.S, s.regime, mdp.sim_params, rng)
   end

4. COMPUTE NEW GREEKS
   τ_new = s.τ - mdp.sim_params.dt
   σ_current = get_vol(mdp.sim_params, regime_new)  # look up vol for current regime
   δ_option_new = bs_delta(S_new, σ_current, mdp.contract, τ_new) * mdp.n_options
   Γ_new = bs_gamma(S_new, σ_current, mdp.contract, τ_new) * mdp.n_options
   δ_net_new = δ_option_new - new_hedge_shares
   m_new = S_new / mdp.contract.K

5. COMPUTE REWARD
   # Option value change
   V_old = bs_price(s.S, get_vol(mdp.sim_params, s.regime), mdp.contract, s.τ) * mdp.n_options
   V_new = bs_price(S_new, σ_current, mdp.contract, τ_new) * mdp.n_options
   ΔV_option = V_new - V_old

   # Hedge P&L
   ΔV_hedge = new_hedge_shares * (S_new - s.S)

   # Net P&L
   ΔPnL = ΔV_option + ΔV_hedge

   # Reward
   r = ΔPnL - txn_cost - mdp.λ * ΔPnL^2

6. BUILD NEW STATE
   sp = HedgingState(δ_net_new, Γ_new, m_new, τ_new, regime_new, S_new, new_hedge_shares)

7. RETURN
   return (sp=sp, r=r)
```

### Critical Sign Convention

Think carefully about signs. If you're long a call option (standard case):
- The option has positive delta (its value goes up when the stock goes up)
- To hedge, you SHORT (sell) shares of the underlying → hedge_shares will be negative
- `δ_net = δ_option - hedge_shares` → if you've hedged perfectly, δ_net ≈ 0
- When the stock goes up: option gains value (good), hedge loses value (expected), net ≈ 0

**Test this with a simple case**: Long 1 ATM call, perfectly hedged (hedge_shares = BS delta), stock goes up 1%. Net P&L should be close to zero (just the second-order gamma term).

---

## 3.2 Value Iteration

**ADM Reference:** Ch. 7.5, Algorithm 7.6.

### Pseudocode

```
INPUT: HedgingMDP, StateGrid, max_iterations, tolerance
OUTPUT: ValueIterationPolicy (V, Q, policy)

1. INITIALIZE
   V = zeros(n_states)          # Value function
   Q = zeros(n_states, n_actions)
   policy = ones(Int, n_states)  # Default: action 1 (no hedge)

2. ITERATE
   for iter = 1:max_iterations
       V_old = copy(V)

       for si = 1:n_states
           s = index_to_state(si, grid)

           if isterminal(mdp, s)
               V[si] = 0.0      # Terminal value
               continue
           end

           for ai = 1:n_actions
               # ESTIMATE Q(s,a) using Monte Carlo sampling
               # Because transitions are stochastic (random price moves),
               # we can't enumerate all next states exactly.
               # Instead, sample N transitions and average.
               total_value = 0.0
               N_samples = 50   # Number of MC samples per (s,a) pair

               for _ = 1:N_samples
                   (sp, r) = @gen(:sp, :r)(mdp, s, ai, rng)
                   sp_idx = state_to_index(sp, grid)
                   total_value += r + mdp.γ_discount * V_old[sp_idx]
               end

               Q[si, ai] = total_value / N_samples
           end

           V[si] = maximum(Q[si, :])
           policy[si] = argmax(Q[si, :])
       end

       # CHECK CONVERGENCE
       residual = maximum(abs.(V - V_old))
       println("Iteration $iter, Bellman residual: $residual")
       if residual < tolerance
           println("Converged after $iter iterations.")
           break
       end
   end

3. RETURN ValueIterationPolicy(V, Q, policy, grid)
```

### Important Notes

**Monte Carlo value iteration:** Classical value iteration enumerates all possible next states and weights by their transition probability: Q(s,a) = Σ_{s'} T(s'|s,a) [R(s,a,s') + γV(s')]. But your transitions are continuous (GBM generates any real-valued price), so you can't enumerate them. Instead, you sample N transitions from the generative model and average. This is standard for continuous-state MDPs — ADM Ch. 8 discusses function approximation approaches, but sampling-based VI on a discretized grid is simpler and fine for your state space size.

**N_samples tradeoff:** More samples = more accurate Q estimates but slower. Start with N_samples=30 to get things working, then increase to 100+ for final results. With 20,000 states × 5 actions × 50 samples, that's 5 million gen() calls per iteration. Julia should handle this in seconds.

**Parallelism:** The inner loop over states is embarrassingly parallel. If it's too slow, add `Threads.@threads` to the state loop. But get it working single-threaded first.

---

## 3.3 DQN Training Loop

**ADM Reference:** Ch. 17.2 (Q-Learning), 17.6 (Action Value Function Approximation), 17.7 (Experience Replay).

### Pseudocode

```
INPUT: HedgingMDP, DQNParams
OUTPUT: Trained DQNAgent

1. INITIALIZE
   Create Q_online neural network: Dense(state_dim, 64, relu) → Dense(64, 64, relu) → Dense(64, n_actions)
   Create Q_target as a deepcopy of Q_online
   Create empty ReplayBuffer with capacity = params.buffer_size
   optimizer = Adam(params.learning_rate)
   step_count = 0

2. TRAINING LOOP
   for episode = 1:params.n_episodes
       s = rand(initialstate(mdp))
       episode_reward = 0.0

       while !isterminal(mdp, s)
           # SELECT ACTION (ε-greedy)
           ε = linear_decay(step_count, params.ε_start, params.ε_end, params.ε_decay_steps)
           if rand(rng) < ε
               a = rand(rng, actions(mdp))
           else
               s_vec = state_to_vector(s)
               q_values = Q_online(s_vec)
               a = argmax(q_values)
           end

           # STEP ENVIRONMENT
           result = @gen(:sp, :r)(mdp, s, a, rng)
           sp, r = result.sp, result.r
           done = isterminal(mdp, sp)

           # STORE EXPERIENCE
           store_experience!(buffer, state_to_vector(s), a, r, state_to_vector(sp), done)

           # TRAIN (if buffer has enough samples)
           if buffer.size >= params.batch_size
               batch = sample_batch(buffer, params.batch_size, rng)

               # Compute targets
               # For each (s, a, r, sp, done) in batch:
               #   if done: target = r
               #   else:    target = r + γ * max_a' Q_target(sp, a')

               # Compute loss = mean((Q_online(s)[a] - target)^2)

               # Gradient step on Q_online
               grads = Flux.gradient(Flux.params(Q_online)) do
                   # loss computation
               end
               Flux.Optimise.update!(optimizer, Flux.params(Q_online), grads)
           end

           # UPDATE TARGET NETWORK
           step_count += 1
           if step_count % params.target_update_freq == 0
               Q_target = deepcopy(Q_online)
           end

           s = sp
           episode_reward += r
       end

       # LOG PROGRESS
       if episode % 100 == 0
           println("Episode $episode, reward: $episode_reward, ε: $ε")
       end
   end

3. RETURN DQNAgent
```

### Julia/Flux Idiom for the Loss

```julia
function train_step!(agent, rng)
    batch = sample_batch(agent.buffer, agent.params.batch_size, rng)

    # batch.states is a Matrix: (state_dim × batch_size)
    # batch.next_states same shape
    # batch.actions is Vector{Int} of length batch_size
    # batch.rewards is Vector{Float64}
    # batch.dones is Vector{Bool}

    # Compute targets (no gradient needed for target network)
    q_next = agent.Q_target(batch.next_states)          # (n_actions × batch_size)
    max_q_next = maximum(q_next, dims=1)[:]              # (batch_size,)
    targets = batch.rewards .+ agent.params.γ .* (1.0 .- batch.dones) .* max_q_next

    # Gradient step
    gs = gradient(Flux.params(agent.Q_online)) do
        q_all = agent.Q_online(batch.states)             # (n_actions × batch_size)
        # Index into q_all to get Q(s,a) for the action actually taken
        q_sa = [q_all[batch.actions[i], i] for i in 1:length(batch.actions)]
        return Flux.mse(q_sa, targets)
    end
    Flux.Optimise.update!(agent.optimizer, Flux.params(agent.Q_online), gs)
end
```

**Note on Flux version:** The exact gradient API may differ slightly between Flux versions. Check which version you have installed and consult the Flux.jl docs if the gradient call syntax doesn't match.

---

## 3.4 Exact Bayesian Belief Update

**ADM Reference:** Ch. 19.2 (Discrete State Filter), Algorithm 19.1.

### Pseudocode

```
INPUT:  current belief b (Vector{Float64}, length = n_regimes)
        observed log-return r_obs (Float64)
        RegimeSwitchingParams (contains transition matrix T, σ levels, dt)
OUTPUT: updated belief b' (Vector{Float64})

1. PREDICT (propagate through transition model)
   b_pred = T' * b
   # T' is the TRANSPOSE of the transition matrix
   # b_pred[j] = Σ_i T[i,j] * b[i] = probability of being in regime j after transition

2. UPDATE (incorporate observation)
   for j = 1:n_regimes
       σ_j = params.σ_levels[j]
       # Observation likelihood: probability of seeing this return under regime j
       likelihood_j = pdf(Normal(params.μ * params.dt, σ_j * sqrt(params.dt)), r_obs)
       b_new[j] = likelihood_j * b_pred[j]
   end

3. NORMALIZE
   b_new ./= sum(b_new)

4. RETURN b_new
```

**This is 5 lines of Julia code.** That's the beauty of the exact filter for discrete hidden states. Implement this first, then build the particle filter and verify they agree.

---

## 3.5 Particle Filter

**ADM Reference:** Ch. 19.6 (Particle Filter), Algorithm 19.6.

### Pseudocode

```
INPUT:  particles (Vector{Int}, length = N, each is a regime index)
        weights (Vector{Float64}, length = N)
        observed log-return r_obs
        RegimeSwitchingParams
OUTPUT: updated particles and weights

1. PROPAGATE
   for i = 1:N
       # Sample new regime for particle i from transition distribution
       old_regime = particles[i]
       transition_probs = params.transition_matrix[old_regime, :]
       particles[i] = sample(Categorical(transition_probs), rng)
   end

2. REWEIGHT
   for i = 1:N
       σ_i = params.σ_levels[particles[i]]
       weights[i] *= pdf(Normal(params.μ * params.dt, σ_i * sqrt(params.dt)), r_obs)
   end
   weights ./= sum(weights)     # Normalize

3. RESAMPLE (if needed)
   # Effective sample size
   ESS = 1.0 / sum(weights .^ 2)
   if ESS < N / 2
       # Systematic resampling
       particles, weights = systematic_resample(particles, weights, rng)
   end

4. RETURN particles, weights
```

### Converting Particles to Belief Vector

```julia
function particles_to_belief(particles::Vector{Int}, weights::Vector{Float64}, n_regimes::Int)
    belief = zeros(n_regimes)
    for i in eachindex(particles)
        belief[particles[i]] += weights[i]
    end
    return belief  # Already normalized since weights sum to 1
end
```

---

# TESTING STRATEGY

Testing is organized by module. **The rule: never move to the next module until the current one's tests pass.**

## Test Module 1: BlackScholes

```
TEST 1.1: Put-call parity
  bs_price(S, σ, call) - bs_price(S, σ, put) ≈ S*exp(-qT) - K*exp(-rT)
  Use several (S, K, σ, T) combinations. Tolerance: < 1e-10.

TEST 1.2: Delta bounds
  For calls: 0 ≤ bs_delta ≤ 1 for all valid inputs.
  For puts: -1 ≤ bs_delta ≤ 0.

TEST 1.3: Gamma symmetry
  Gamma should be the same for calls and puts with same parameters.

TEST 1.4: ATM delta near 0.5
  bs_delta(S=100, K=100, σ=0.2, τ=0.25, call) ≈ 0.5 (within ~0.05).

TEST 1.5: Limits
  As τ → 0: ITM call delta → 1, OTM call delta → 0.
  As σ → 0: same limiting behavior.
```

## Test Module 2: MarketSim

```
TEST 2.1: GBM non-negativity
  Run 10,000 steps. All prices must be > 0.

TEST 2.2: GBM distributional check
  Generate 100,000 single-step returns.
  Mean should be ≈ μ*dt (within sampling error).
  Std should be ≈ σ*√dt (within sampling error).

TEST 2.3: Regime switching stays in bounds
  Run 10,000 steps. Regime should always be in {1, 2, ..., n_regimes}.

TEST 2.4: Regime transition frequency
  Run 100,000 steps. Empirical transition frequencies should
  approximately match the transition matrix.
```

## Test Module 3: HedgingEnv

```
TEST 3.1: Perfect hedge, zero cost
  Set κ=0, hedge 100% every step. Net P&L each step should be ≈ 0
  (up to gamma/second-order effects). Verify |ΔPnL| < some small bound.

TEST 3.2: No hedge baseline
  Action = 0% every step. Cumulative P&L should equal option payoff
  minus initial option value (unhedged position).

TEST 3.3: Terminal state
  After τ reaches 0, isterminal returns true.

TEST 3.4: Transaction cost accounting
  Set κ=0.01. Hedge 100% on step 1. Verify txn cost =
  κ * |shares_traded| * S, and that it appears in the reward.

TEST 3.5: State discretization roundtrip
  For random continuous states, state_to_index → index_to_state
  should return a state "close to" the original (within bin width).
```

## Test Module 4: Solvers

```
TEST 4.1 (Value Iteration): Zero txn cost convergence
  With κ=0, the VI policy should converge to "always hedge 100%"
  because there's no cost to hedging. This verifies the reward
  function and transitions are correct.

TEST 4.2 (Value Iteration): High txn cost
  With very high κ, the VI policy should converge to "never hedge"
  (action 0%) because hedging costs more than the risk.

TEST 4.3 (DQN): Overfit a tiny problem
  Reduce state space to something trivial (e.g., 2 timesteps).
  DQN should learn to match the VI policy.

TEST 4.4 (MCTS): Smoke test
  Run MCTS on the HedgingMDP, verify it returns valid actions
  and doesn't crash.

TEST 4.5 (Belief updater): Known regime
  Feed the exact Bayesian updater a long sequence of returns
  drawn from regime 1. After enough observations, belief[1] ≈ 1.0.

TEST 4.6 (Particle filter): Convergence to exact
  Run both exact and particle filter on the same observation sequence.
  With 10,000 particles, beliefs should match within 0.01.
```

---

# INTEGRATION ORDER & CHECKPOINTS

This is the exact sequence of what to build and when to stop and validate.

## Phase 1: Foundation (Days 1-3)

```
Step 1: black_scholes.jl
   Build → Run tests 1.1-1.5 → CHECKPOINT: all pass ✓

Step 2: market_sim.jl (GBM only)
   Build → Run tests 2.1-2.2 → CHECKPOINT: all pass ✓

Step 3: hedging_env.jl (GBM/Level 1 only, no regime switching)
   Build HedgingState, HedgingMDP, gen() function
   Run tests 3.1-3.4 → CHECKPOINT: gen() produces sensible rewards ✓

Step 4: baselines.jl
   Build BS hedge policy
   Manually run 1 episode with BS hedge, print each step
   Verify P&L is near zero → CHECKPOINT ✓
```

## Phase 2: Level 1 Complete (Days 4-7)

```
Step 5: discretization.jl
   Build StateGrid, state_to_index, index_to_state
   Run test 3.5 → CHECKPOINT: roundtrip works ✓
   Run simulation to calibrate bin ranges empirically.

Step 6: value_iteration.jl
   Build → Run tests 4.1, 4.2
   CHECKPOINT: VI converges, policy matches intuition ✓

Step 7: evaluation.jl (basic version)
   Build run_episode, evaluate_policy, compare_policies
   Run VI policy vs BS baseline on 10,000 episodes
   Generate P&L distribution plots
   CHECKPOINT: VI outperforms BS when κ > 0 ✓

   >>> LEVEL 1 IS DONE. Everything from here is additive. <<<
```

## Phase 3: Level 2 (Days 8-14)

```
Step 8: market_sim.jl (add regime switching)
   Add RegimeSwitchingParams, step_regime
   Run tests 2.3-2.4 → CHECKPOINT ✓

Step 9: hedging_env.jl (add regime to state)
   Update gen() to handle RegimeSwitchingParams
   Test: same as 3.1-3.4 but with regime switching
   CHECKPOINT: gen() works with regimes ✓

Step 10: mcts_interface.jl
   Build thin wrapper, run test 4.4
   Run MCTS on regime-switching MDP, compare to BS
   CHECKPOINT: MCTS produces reasonable policies ✓

Step 11: dqn.jl
   Build DQN agent, replay buffer, training loop
   Run test 4.3 (overfit tiny problem)
   Train on full regime-switching MDP
   CHECKPOINT: DQN training curve shows improvement ✓
   (If DQN is not working by day 12, deprioritize and focus on
    MCTS + VI comparison. A solid 2-method comparison is better
    than a broken 3-method one.)

Step 12: Level 2 experiments
   Three-way comparison: MCTS vs DQN vs BS
   Per-regime analysis
   CHECKPOINT: clear, plotted results ✓

   >>> LEVEL 2 IS DONE. <<<
```

## Phase 4: Level 3 (Days 15-20)

```
Step 13: belief_updaters.jl
   Build exact Bayesian updater → test 4.5
   Build particle filter → test 4.6
   CHECKPOINT: both updaters work, PF matches exact ✓

Step 14: hedging_env.jl (POMDP version)
   Build HedgingPOMDP, gen() with observations
   Test: same as MDP tests but verify observation is returned

Step 15: qmdp.jl
   Run VI once per regime to get Q tables
   Build QMDP policy that averages using belief
   Test: with belief = [1.0, 0.0], QMDP should match regime-1 VI policy
   CHECKPOINT ✓

Step 16: pomdp_interface.jl (POMCPOW)
   Build thin wrapper
   Run on HedgingPOMDP
   CHECKPOINT: runs without crashing, produces policies ✓

Step 17: Level 3 experiments
   Compare: QMDP vs POMCPOW vs Level-2-MDP vs BS
   Cost of partial observability analysis
   Belief ambiguity analysis

   >>> LEVEL 3 IS DONE. <<<
```

## Phase 5: Paper (Days 21-30)

```
Step 18: generate_figures.jl
   Polish all plots for paper quality
   Ensure consistent styling, axis labels, legends

Step 19: Write paper
   Follow the structure from the outline discussion:
   Intro → Formulation → Methods → Experiments → Results → Conclusion

Step 20: Review and submit
```

---

# JULIA IDIOMS & TIPS

Since this is your first Julia project, here are patterns you'll use repeatedly:

## Structs are Immutable by Default (and That's Good)

```julia
struct HedgingState        # Immutable — fields cannot change after construction
    δ_net::Float64
    # ...
end

# To "modify" a state, create a new one:
new_state = HedgingState(new_delta, new_gamma, ...)
```

Use `mutable struct` only when you genuinely need to mutate in place (e.g., ReplayBuffer, BeliefUpdater).

## Multiple Dispatch (Julia's Superpower)

```julia
# Same function name, different behavior based on argument types:
step_price(S, params::GBMParams, rng) = ...         # GBM version
step_price(S, regime, params::RegimeSwitchingParams, rng) = ...  # Regime version

# Julia picks the right one automatically based on the type of params.
```

## RNG Discipline

Always pass an `rng::AbstractRNG` explicitly. Never call `rand()` without it. This makes your results reproducible.

```julia
using Random
rng = MersenneTwister(42)   # Fixed seed
z = randn(rng)              # Reproducible
```

## Named Tuples for gen()

POMDPs.jl expects `gen` to return a NamedTuple:

```julia
function POMDPs.gen(mdp::HedgingMDP, s::HedgingState, a::Int, rng::AbstractRNG)
    # ... compute sp and r ...
    return (sp=sp, r=r)
end
```

## Broadcasting with Dots

```julia
weights ./= sum(weights)       # In-place normalize
targets = rewards .+ γ .* max_q   # Element-wise operations
```

## Type Annotations Help Performance

Julia is fast when the compiler knows the types. Type-annotate struct fields (you're already doing this). For function arguments, it's optional for correctness but helps with dispatch:

```julia
function bs_delta(S::Float64, σ::Float64, contract::OptionContract, τ::Float64)::Float64
```
