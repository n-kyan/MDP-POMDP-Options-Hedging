# Change History

---

## Session 1 — Core infrastructure (Modules 1–8)

Completed before this log was started. Modules 1–8 implemented and passing tests. See git log.

---

## Session 2 — Module 9: MCTS+DPW oracle MDP

**What**: Implemented `src/9_mcts_mdp.jl` — an oracle MDP where the agent sees the true Hardy regime index at every step. Uses MCTS with Double Progressive Widening (DPWSolver).

**Key decisions**:
- `MDPState` includes `regime_idx` (fully observable)
- `gen()` prices options at `V_market = bs_all_belief_weighted(...)` — exact transition-row-weighted BS, not Jensen approximation
- Wealth delta uses `V_market` both before and after — correct reward signal
- `OracleGLFTRollout` uses `oracle_σ = sqrt(Σ pᵢⱼ σⱼ²)` for rollout actions
- `evaluate_mcts_mdp` replans at every step (keep_tree=false) and syncs MDPState from true env

**Smoke test result (5 ep)**: mean P&L ≈ +2.37

---

## Session 2 — Module 10: POMCPOW POMDP

**What**: Implemented `src/10_pomcpow.jl` — the core scientific experiment. Agent uses log-vol random walk as generative model (NOT Hardy). Infers σ from particle belief.

**Bug sequence fixed**:

1. `InexactError: Int64(-Inf)` — POMCPOW computes `ceil(Int, log(eps)/log(discount))` to cap tree depth. With `discount=1.0`, `log(1.0)=0` → division by zero → `-Inf` → crash. Fixed by setting `discount=0.9999`.

2. `no method matching actions(::OptionsMM_POMDP)` — `next_action` was dispatched as `(pomdp, b, h, old)` but POMCPOW calls `(sampler_object, pomdp, b, h)`. Fixed by creating `POWActionSampler` struct with correct 4-arg dispatch.

3. `no method matching extract_belief(::BootstrapFilter, ...)` — `BasicPOMCP.RolloutEstimator` internally calls `extract_belief(updater, tree_node)` which is only defined for BasicPOMCP's own updater types. Fixed by switching to `BasicPOMCP.FORollout(GLFTStateRollout)` — a state-based rollout that doesn't require belief extraction.

4. `σ_hat = 3051` (exploding vol estimate) — `BootstrapFilter.update()` returns **unnormalized** likelihood weights when ESS is high enough that resampling doesn't trigger. `belief_mean_σ` was computing `Σ w_i σ_i` with weights summing to ~979. Fixed by dividing by `sum(ws)` in both `belief_mean_σ` and the Rao-Blackwellization step.

5. Catastrophic P&L (-7000) from belief desynchronization — after each true-env step, fills are asymmetric (against V_market) while the agent's gen() assumes symmetric fills (against hat_V). Particles diverged from true portfolio after a few steps. Fixed with Rao-Blackwellization: after each `POMDPs.update()`, snap all fully-observable particle components (S, q, q_spot, cash, τ, K, options_completed) to true values.

**Current performance (20 ep)**: mean P&L = -19.77, std = 13.22, sharpe = -1.50

---

## Session 2 — Oracle benchmark bug fix

**What**: All four analytical benchmarks were using `σ_hat ≈ 0.20` (particle filter prior) for the quote midpoint instead of oracle σ.

**Root cause**: `step_environment!` used the particle filter's `get_σ_hat(pf)` for `hat_V` computation. Benchmarks were passing the oracle σ to the spread formula but not to the midpoint.

**Fix**: Added `σ_hat_override::Float64 = NaN` keyword to `step_environment!`. When non-NaN, overrides both the hat_V computation and the next agent state σ_hat. Benchmark runner passes `σ_hat_override = oracle_σ(env)` when `use_oracle=true`.

**Result**: GLF-T+WW Hardy: -2.49 → +4.87 after fix.

---

## Session 3 — POMCPOW performance investigation

**Diagnostic added**: `diagnose_pomcpow()` in `src/10_pomcpow.jl` prints per-step: true_σ, σ_hat, δ_pomcpow, δ_glft (at same σ_hat), fill_dir, q_after. Computes per-episode mean δ_ratio and pct_narrower.

**Results (5 ep diagnostic + 30 ep MCTS run)**:

MCTS+DPW oracle (30 ep):
- mean = +2.02, std = 2.78, sharpe = 0.726
- All positive, confirms MDP gen() and shared reward function are correct

POMCPOW δ vs GLF-T δ (5 ep):
- Mean δ_ratio ≈ 2.6 (POMCPOW is on average wider than GLF-T)
- But ~50% of individual steps POMCPOW is NARROWER than GLF-T
- Spread is highly erratic within each episode (e.g., 0.09 to 4.77 in same ep)
- The critical pattern: on steps where fills occur, POMCPOW often happens to quote narrower → adverse selection losses

**Diagnosis**: The POMCPOW losses are not simple model misspecification. The planning fails to produce consistently good spreads because:
1. Action sampling is uniform over [δ_lo, hat_V] — no structure guiding the tree toward GLF-T-like values
2. The gen() reward uses perceived wealth (hat_V ≠ V_market) — tree optimizes for a systematically wrong objective when σ_hat ≠ true_σ
3. With 50 queries and depth 5, the tree can't reliably distinguish narrow from wide spreads because symmetric-fill gen() makes both look profitable in the agent's model

**Status**: Under investigation. No solver logic changed yet.
