# ============================================================================
# animate_sim.jl — Animated Price Path Visualization
# ============================================================================
# Generates a GIF showing the price path being drawn step by step.
# For regime-switching models, the line color changes with the regime.
#
# Usage:
#   julia animate_sim.jl
#
# Output:
#   price_path_constant.gif   — constant vol simulation
#   price_path_regimes.gif    — regime-switching simulation
#
# Dependencies:
#   Pkg.add("Plots")
#   Pkg.add("Distributions")
# ============================================================================

include("market_sim.jl")   # Load the simulation module

using Plots

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

# Simulation parameters
S0 = 100.0          # Starting price
r  = 0.05           # Risk-free rate (5% annualized)
dt = 1 / 252        # Daily timesteps
n_steps = 504       # 2 years of trading days

# Regime-switching parameters (calibrated to S&P 500, Cerboni Baiardi et al. 2020)
σ_low  = 0.10       # Low regime:    ~10% annualized vol (VIX typically < 17)
σ_mid  = 0.20       # Medium regime: ~20% annualized vol (VIX roughly 17–28)
σ_high = 0.40       # High regime:   ~40% annualized vol (VIX > 28, crisis)
T_matrix = [0.977  0.023  0.000;   # low    → low 97.7%, medium 2.3%, high 0.0%
            0.027  0.967  0.006;   # medium → low 2.7%,  medium 96.7%, high 0.6%
            0.000  0.027  0.973]   # high   → low 0.0%,  medium 2.7%, high 97.3%

# Animation settings
frames_per_step = 1     # 1 frame per timestep (increase for smoother but larger GIF)
fps = 120                # Playback speed: 30 frames per second
                        # At fps=30 and 252 steps, the animation takes ~8.4 seconds

# Regime colors
regime_colors = Dict(
    1 => :royalblue,    # Low vol regime
    2 => :darkorange,   # Medium vol regime
    3 => :crimson       # High vol / crisis regime
)

# ────────────────────────────────────────────────────────────────────────────
# Helper: Build animation from a SimulationResult
# ────────────────────────────────────────────────────────────────────────────

"""
    animate_price_path(result; title, fps, filepath)

Create an animated GIF of a price path being drawn step by step.
For regime-switching simulations, line segments are colored by regime.
"""
function animate_price_path(
    result::SimulationResult;
    title::String = "Simulated Price Path",
    fps::Int = 30,
    filepath::String = "price_path.gif"
)
    prices = result.prices
    regimes = result.regimes
    n = length(prices) - 1   # number of steps

    # Compute axis limits with padding so the plot doesn't jump around
    price_min = minimum(prices) * 0.97
    price_max = maximum(prices) * 1.03

    # Time axis: trading days
    days = 0:n

    # Determine if this is a regime-switching simulation
    has_regimes = any(r != 1 for r in regimes)

    println("Generating animation: $filepath ($n frames)...")

    anim = @animate for frame in 1:n
        # Base plot with fixed axes
        p = plot(
            xlim = (0, n),
            ylim = (price_min, price_max),
            xlabel = "Trading Day",
            ylabel = "Spot Price (\$)",
            title = title,
            legend = has_regimes ? :topright : false,
            size = (900, 500),
            grid = true,
            gridstyle = :dash,
            gridalpha = 0.3,
            background_color = :white,
            fontfamily = "Computer Modern"
        )

        if has_regimes
            # Draw line segments colored by regime
            for t in 1:frame
                seg_color = get(regime_colors, regimes[t + 1], :gray)
                plot!(p,
                    [t - 1, t],
                    [prices[t], prices[t + 1]],
                    color = seg_color,
                    linewidth = 1.5,
                    label = false
                )
            end

            # Add regime legend entries (only once, using invisible scatter points)
            scatter!(p, [], [], color = regime_colors[1], label = "Low vol (σ = $(σ_low))", markersize = 0)
            scatter!(p, [], [], color = regime_colors[2], label = "Medium vol (σ = $(σ_mid))", markersize = 0)
            scatter!(p, [], [], color = regime_colors[3], label = "High vol (σ = $(σ_high))", markersize = 0)

            # Add a dot at the current price
            current_color = get(regime_colors, regimes[frame + 1], :gray)
            scatter!(p, [frame], [prices[frame + 1]],
                color = current_color,
                markersize = 6,
                label = false
            )
        else
            # Constant vol: single color line
            plot!(p,
                0:frame,
                prices[1:frame + 1],
                color = :royalblue,
                linewidth = 1.5,
                label = false
            )

            # Current price dot
            scatter!(p, [frame], [prices[frame + 1]],
                color = :royalblue,
                markersize = 6,
                label = false
            )
        end

        # Annotation: current price and day
        annotate!(p,
            frame,
            prices[frame + 1],
            text("\$$(round(prices[frame + 1], digits=2))", 8, :left, :bottom)
        )
    end

    gif(anim, filepath, fps = fps)
    println("Animation saved to: $filepath")
end

# ────────────────────────────────────────────────────────────────────────────
# Generate Animations
# ────────────────────────────────────────────────────────────────────────────

println("=" ^ 60)
println("Spot Price Simulation — Animation Generator")
println("=" ^ 60)

# --- Constant Volatility (Level 1) ---
println("\n--- Constant Volatility (σ = 0.20) ---")
params_const = MarketParams(S0, r, dt, ConstantVol(0.20))
result_const = simulate_path(params_const, n_steps; seed=67)

println("  Start price: \$$(result_const.prices[1])")
println("  End price:   \$$(round(result_const.prices[end], digits=2))")
println("  Min price:   \$$(round(minimum(result_const.prices), digits=2))")
println("  Max price:   \$$(round(maximum(result_const.prices), digits=2))")

animate_price_path(result_const;
    title = "GBM Price Path — Constant Vol (σ = 20%)",
    fps = fps,
    filepath = "price_path_constant.gif"
)

# --- Regime Switching (Level 2+) ---
println("\n--- Regime Switching (σ_low=$σ_low, σ_mid=$σ_mid, σ_high=$σ_high) ---")
vol_rs = RegimeSwitchingVol([σ_low, σ_mid, σ_high], T_matrix, 1)
params_rs = MarketParams(S0, r, dt, vol_rs)
result_rs = simulate_path(params_rs, n_steps; seed=67)

# Count days in each regime
n_low    = count(==(1), result_rs.regimes)
n_mid    = count(==(2), result_rs.regimes)
n_high   = count(==(3), result_rs.regimes)
println("  Start price:       \$$(result_rs.prices[1])")
println("  End price:         \$$(round(result_rs.prices[end], digits=2))")
println("  Days in low vol:   $n_low  ($(round(100*n_low/(n_steps+1), digits=1))%)")
println("  Days in medium vol: $n_mid ($(round(100*n_mid/(n_steps+1), digits=1))%)")
println("  Days in high vol:  $n_high  ($(round(100*n_high/(n_steps+1), digits=1))%)")

animate_price_path(result_rs;
    title = "GBM Price Path — Regime Switching Vol",
    fps = fps,
    filepath = "price_path_regimes.gif"
)

println("\n" * "=" ^ 60)
println("Done! Check the .gif files in your working directory.")
println("=" ^ 60)