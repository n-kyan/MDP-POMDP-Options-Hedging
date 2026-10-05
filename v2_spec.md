# Options Market Making Lab — v2 Spec

*Draft, 2026-10-05. Items marked **(proposed)** came up in discussion but have not been confirmed yet.*

---

## 1. Thesis

> **How does an options market maker make money, where exactly does that P&L come from, and how do those sources shift under different market conditions?**

This is a **P&L attribution study**, not a strategy-optimization project. A textbook market maker trades in a textbook simulated market. The work is to break its P&L down into its real sources and see how they move as market conditions change.

**Why a simulation (proposed framing):** in a simulator the true drivers of P&L are known. That makes it possible to *validate* an attribution method, which can't be done on real data because nobody knows the ground truth. The simulator is a controlled lab, not a world to "beat."

**Who it's for:** me, to understand options market making from first principles, and quant trading employers as a portfolio piece. P&L explain is real daily desk work, and the project should show trader-style reasoning in expected values, edge vs. risk, and statistical rigor.

## 2. Guiding principles

1. **Lean full pipeline first.** Get the simplest version of *every* stage working end to end before deepening any one stage.
2. **Mainstream literature only.** The market and the market maker both follow established quant-finance and options-market-making practice. Nothing novel and nothing exotic.
3. **No reinforcement learning.** The market maker is rule-based and interpretable. Every decision should be defensible on a whiteboard.
4. **A feature earns its place by creating or isolating a P&L term (proposed).** If adding something doesn't change the attribution table (§5), it's out of scope.
5. **Assumptions are justified, not picked.** Each design choice is written down with its reason and a literature reference.
6. **Profitability is a result, not an assumption.** When and why the market maker *loses* is part of the analysis.

## 3. The world (minimal version)

| Component | Minimal version | Why it's needed |
|---|---|---|
| **Spot** | One simulated asset | The underlying and the hedge instrument |
| **Volatility** | Non-constant; realized vol can differ from implied vol | Without that gap, the gamma/theta and vega terms vanish |
| **Options** | 4 European options in a 2×2 grid: {low K, high K} × {short T, long T} | The minimum for a surface with level, skew and term structure, and for a multi-instrument book |
| **Fair value** | A market-consensus implied-vol surface over the 4 options; theo = Black-Scholes at surface vol (proposed) | Defines "fair," which fills and edge are measured against |
| **Order flow** | Customer orders arrive at random; fill probability falls as the quote moves away from theo | The mechanism that turns quotes into inventory |
| **Adverse selection** | **Not in the baseline.** Flow is uninformed | Not assumed. It's added later as a scenario |

Open decisions: the spot/vol process, how the surface moves, the flow model's functional form, the timestep, and whether option expiries roll.

## 4. The market maker (minimal version)

- **Theo:** prices each option off the surface. In the baseline it has no informational disadvantage.
- **Quotes:** theo ± a half-width.
- **Book-aware skew:** shifts the quotes on all 4 options according to the book's *aggregate* Greeks, so it leans away from exposures it already holds (proposed: delta and vega).
- **Hedging:** takes liquidity in spot to delta-hedge, inside a band; pays a spot spread or cost.
- **Parameters:** a handful of interpretable numbers (width, skew strength, hedge band). No optimization loop.

## 5. P&L attribution: the core deliverable

| Term | Driver |
|---|---|
| Edge / spread capture | Trade price − theo at the time of the trade |
| Gamma + theta | ≈ ½ Γ S² (σ²_realized − σ²_implied) dt |
| Vega | ν · Δσ_implied |
| Delta (residual) | Unhedged delta × ΔS |
| Hedging cost | Cost of crossing the spot spread |
| Residual | Higher-order Greeks and discretization; its size measures attribution quality |

**Validation (proposed):** the attribution must sum to realized P&L with a small, understood residual, and must recover terms that are known to be present. Markouts (how theo moves after a fill) should average ≈ 0 in the uninformed baseline.

## 6. Scenarios (proposed, fixed list)

Each scenario is designed to stress one row of §5. Anything not on this list goes to a "later" file.

1. **Baseline:** realized ≈ implied
2. **Realized > implied:** gamma pain
3. **Implied-vol shock:** vega
4. **One-sided flow:** inventory builds up
5. *(optional)* **Informed flow:** adverse selection shows up in markouts

## 7. Pipeline

```
simulate world + MM  →  trade/position log  →  attribution  →  analysis & figures  →  report
```

**Build order (proposed):** every milestone ends with something showable.

| Milestone | Scope | Artifact |
|---|---|---|
| **M1: walking skeleton** | Simplest version of every stage, end to end | An attribution table for one run |
| **M2: vol separation** | Realized ≠ implied | Gamma/theta term appears and is validated |
| **M3: moving surface** | Surface dynamics | Vega term appears and is validated |
| **M4: scenarios** | Run §6 with many seeds | A scenario-comparison report with confidence intervals |
| **M5: optional** | Informed flow | Adverse-selection / markout analysis |

## 8. Definition of done

1. Attribution reconciles to total P&L with a small residual and passes validation.
2. A report shows how each P&L term shifts across the scenarios, with confidence intervals.
3. The whole pipeline reproduces from a single command.
4. The README explains every assumption and its justification.

## 9. Out of scope

Reinforcement learning or any learned policy · strategy optimization or tuning for profit · more than 4 options (until done) · real market data · agent-based market simulation · multiple underlyings · American options, dividends, rates modeling · order-book microstructure / queue modeling · latency and execution realism

## 10. Tech & repo

- **Language:** Python throughout (simulation and analysis).
- **Location:** a new, separate repository. v1 (this Julia repo) is frozen as-is and used for reference only; no code is ported over.
