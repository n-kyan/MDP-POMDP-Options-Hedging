# Options Hedging MDP/POMDP — Project Tracker

**Course:** CSCI 5264 — Decision Making Under Uncertainty
**Author:** Kyan Nelson
**Start:** March 26, 2026
**Target Submission:** ~May 5, 2026 (confirm exact date)

---

## Milestones

| Milestone | Target Date | Status |
|-----------|------------|--------|
| M0: Environment & infrastructure | April 6 | 🔲 Not started |
| M1: Level 1 — Value iteration baseline | April 13 | 🔲 Not started |
| M2: Level 2 — MCTS + DQN | April 23 | 🔲 Not started |
| M3: Level 3 — POMDP (stretch) | April 28 | 🔲 Not started |
| M4: Paper + submission | May 5 | 🔲 Not started |

---

## Kanban

### ✅ Done
<!-- Move completed tasks here with date -->
- [x] 3/28/26 - Set up Julia project: directory structure, Project.toml, dependencies (POMDPs.jl, Flux.jl)
- [x] 3/28/26 Implement GBM price simulator: single function, takes vol/dt/steps, returns price path
- [x] 3/28/26 Implement BS delta and gamma: closed-form functions, verify against known values

**M0: Environment & Infrastructure**
- [x] 3/28/26 - Define option position struct (strike, expiry, quantity, long/short)
- [x] 3/28/26 - Build portfolio container: holds multiple options, computes aggregate delta/gamma
- [x] 3/28/26 - Implement transaction cost model (proportional cost per hedge trade)
- [x] 3/28/26 - Build single-step environment function: takes state + action, returns next state + reward
- [x] 2/28/26 - Implement BS benchmark hedger: delta-hedge to neutral every step, track P&L and costs
- [x] 2/28/26 - Build episode runner: simulate full path, run a policy, collect cumulative P&L/variance/Sharpe
- [x] 2/28/26 - Verify environment: run BS hedger on constant-vol GBM, confirm P&L variance decreases with hedge frequency. P&L variance actually increased with hedge frequency. Maybe this is because the hedging trades cause pnl to change a lot.
- [x] 2/28/26 - Define MDP state discretization scheme (bins for delta, gamma, vol, time)


### 🔨 In Progress
<!-- Only 1-2 tasks here at a time -->
**M1: Level 1 — Value Iteration**
- [ ] Implement MDP struct conforming to POMDPs.jl interface (states, actions, transitions, reward)
- [ ] Implement tabular value iteration from scratch
- [ ] Run value iteration on single-option constant-vol problem
- [ ] Extract and visualize optimal policy (heatmap: delta vs. time-to-expiry → hedge action)
- [ ] Compare optimal policy vs. BS benchmark: P&L, variance, Sharpe, hedge frequency
- [ ] Sensitivity analysis: vary transaction cost level, show how optimal policy changes
- [ ] Document Level 1 results and figures


### 📋 Up Next
<!-- The next 2-3 tasks you'll pick up -->




### 🗄️ Backlog





**M1: Level 1 — Value Iteration**
- [ ] Implement MDP struct conforming to POMDPs.jl interface (states, actions, transitions, reward)
- [ ] Implement tabular value iteration from scratch
- [ ] Run value iteration on single-option constant-vol problem
- [ ] Extract and visualize optimal policy (heatmap: delta vs. time-to-expiry → hedge action)
- [ ] Compare optimal policy vs. BS benchmark: P&L, variance, Sharpe, hedge frequency
- [ ] Sensitivity analysis: vary transaction cost level, show how optimal policy changes
- [ ] Document Level 1 results and figures

**M2: Level 2 — Regime-Switching + MCTS + DQN**
- [ ] Implement regime-switching price simulator (2 regimes, Markov transitions)
- [ ] Extend environment to multi-option portfolio (aggregate Greeks across positions)
- [ ] Update MDP state to include observable regime label
- [ ] Set up MCTS solver from POMDPs.jl, define generative model interface
- [ ] Run MCTS on regime-switching problem, tune hyperparameters (depth, iterations)
- [ ] Implement DQN: state encoding, replay buffer, training loop (Flux.jl for network)
- [ ] Train DQN on simulated episodes, track learning curve
- [ ] Three-way comparison: MCTS vs. DQN vs. BS on same simulated paths
- [ ] Analyze policy differences across vol regimes (does agent hedge more in high-vol?)
- [ ] Analyze policy differences for positive vs. negative gamma portfolios
- [ ] Document Level 2 results and figures

**M3: Level 3 — POMDP (Stretch)**
- [ ] Define POMDP struct: hidden state = regime, observations = recent returns + trailing realized vol
- [ ] Implement particle filter for belief updates over regime
- [ ] Set up POMCPOW or DESPOT solver from POMDPs.jl
- [ ] Run POMDP solver, compare against Level 2 MDP policy (quantify cost of partial observability)
- [ ] Compare POMDP vs. DQN (planning vs. learning under hidden state)
- [ ] Document Level 3 results and figures

**M4: Paper**
- [ ] Outline paper structure (intro, formulation, methods, results by level, discussion)
- [ ] Write problem formulation section (can reuse/expand proposal)
- [ ] Write simulation environment description
- [ ] Write Level 1 methods + results
- [ ] Write Level 2 methods + results
- [ ] Write Level 3 methods + results (if completed)
- [ ] Write discussion: what did the optimal policies reveal about hedging under uncertainty?
- [ ] Final pass: figures, formatting, references
- [ ] Submit

### 🧊 Icebox (not committed)
<!-- Ideas that would be nice but aren't part of the plan -->

- [ ] More realistic transaction cost model (consult Prof. Brown)
- [ ] Vega hedging actions (hedge vol exposure, not just delta)
- [ ] Historical vol calibration (fit regime-switching params to real VIX data)
- [ ] Continuous action space with policy gradient methods

---

## Decision Log
<!-- Record key decisions and rationale so future-you remembers why -->

| Date | Decision | Rationale |
|------|----------|-----------|
| Mar 26 | Chose options hedging over Markov Game / LOB-POMDP | Strongest career signal, clean MDP→POMDP escalation, accessible to non-finance grader |
| Mar 26 | Fixed proportional transaction costs for now | Keeps scope contained; may upgrade after consulting Prof. Brown |
| Mar 26 | Synthetic data only, no market data | Eliminates data pipeline complexity; focus is on decision-theoretic methods |

---

## Notes
<!-- Scratchpad for ideas, bugs, questions to ask Prof. Sunberg -->

- Confirm exact due date and submission format (paper length? code submission?)
- Ask Sunberg: any preference on POMCPOW vs. DESPOT for Level 3?
- Ask Brown: more realistic transaction cost model for options hedging?
