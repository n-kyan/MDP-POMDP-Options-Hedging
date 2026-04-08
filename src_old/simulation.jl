using Random, Statistics

#=

Simulate GBM price paths using the exact discrete update:
    S(t+dt) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)

    
Returns (prices, log_returns) where:
  - prices:      Matrix of size (n_steps+1, n_paths)  — includes S0 at row 1
  - log_returns: Matrix of size (n_steps,   n_paths)

The Itô correction (-σ²/2) ensures E[S(T)] = S0 * exp(μ*T).

=#


function simulate_gbm(
    S0::Float64,
    μ::Float64,
    σ::Float64,
    dt::Float64,
    n_steps::Int;
    n_paths::Int = 1,
    rng::AbstractRNG = Random.default_rng()
)

    @assert S0 > 0.0 "Initial price must be positive"
    @assert σ > 0.0 "Volatility must be positive"
    @assert dt > 0.0 "Time step must be positive"
    @assert n_steps > 0 "Must have at least one step"

    drift = (μ - 0.5 * σ^2) * dt
    diffuse = σ * sqrt(dt)

    log_returns = drift .+ diffuse .* randn(rng, n_steps, n_paths)
    log_prices = vcat(zeros(1, n_paths), cumsum(log_returns; dims=1))
    prices = S0 .* exp.(log_prices)

    return prices, log_returns
end

function verify_gbm(S0::Float64=100.0, μ::Float64=0.05, σ::Float64=0.2, T::Float64=1.0; n_paths::Int=100_000)
    dt, n_steps = 1/252, round(Int, T*252)
    prices, _ = simulate_gbm(S0, μ, σ, dt, n_steps; n_paths=n_paths)
    
    terminal = prices[end, :]
    theoretical_mean = S0 * exp(μ * T)
    
    println("Theoretical E[S(T)]:  $(round(theoretical_mean, digits=4))")
    println("Simulated mean S(T):  $(round(mean(terminal), digits=4))")
    println("Relative error:       $(round(abs(mean(terminal)/theoretical_mean - 1)*100, digits=3))%")
    println("Simulated realized σ: $(round(std(log.(terminal./S0))/sqrt(T)*100, digits=2))% (target: $(σ*100)%)")
    log_terminal = log.(terminal ./ S0)
    println("Skewness of log returns: $(round(skewness_check(log_terminal), digits=3)) (target: ~0)")
end



# run the verify function only if this file is ran directly
if abspath(PROGRAM_FILE) == @__FILE__
    verify_gbm()
end