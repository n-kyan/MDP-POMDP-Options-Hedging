using Distributions: Normal, logpdf

# ---- Construction ------------------------------------------------

# Initializes a particle filter with a log-normal prior over σ.
# Prior: log(σ) ~ N(log(0.20), 0.5²), which covers [~0.07, ~0.55] at 2σ —
# wide enough to encompass both Hardy regimes (0.121 and 0.269).
function ParticleFilter(n::Int; μ_log_σ::Float64 = log(0.20), σ_log_σ::Float64 = 0.5)
    log_σ = Float32.(μ_log_σ .+ σ_log_σ .* randn(n))
    clamp!(log_σ, log(Float32(0.02)), log(Float32(2.0)))
    log_w = fill(0.0, n)  # log(1/n) + const; uniform after normalization
    return ParticleFilter(log_σ, log_w, n)
end

function reset!(pf::ParticleFilter; μ_log_σ::Float64 = log(0.20), σ_log_σ::Float64 = 0.5)
    pf.log_σ .= Float32.(μ_log_σ .+ σ_log_σ .* randn(pf.n))
    clamp!(pf.log_σ, log(Float32(0.02)), log(Float32(2.0)))
    fill!(pf.log_w, 0.0)
end

# ---- Queries -----------------------------------------------------

# Normalized weights (Float64) from log-space storage.
function get_weights(pf::ParticleFilter)::Vector{Float64}
    lw_max = maximum(pf.log_w)
    w = exp.(pf.log_w .- lw_max)
    return w ./ sum(w)
end

# Weighted mean σ — the agent's point estimate of current volatility.
function get_σ_hat(pf::ParticleFilter)::Float64
    w = get_weights(pf)
    return Float64(sum(w .* exp.(pf.log_σ)))
end

# Effective sample size: 1/Σwᵢ². Falls toward 1 as the filter collapses.
function get_ess(pf::ParticleFilter)::Float64
    w = get_weights(pf)
    return 1.0 / sum(w .^ 2)
end

# ---- Update step -------------------------------------------------

# Sequential importance update: weight each particle by the likelihood of
# observing log-return r_t given that particle's σ.
#
# Observation model (from paper Section 3):
#   Z(o_t | σ*) = N(r_t; (μ - ½σ*²)·dt, σ*²·dt)
#
# Runs systematic resampling + jitter when ESS < n/2.
function update!(pf::ParticleFilter, r_t::Float64, config::SimConfig)
    dt = config.Δt
    μ  = config.r

    for i in 1:pf.n
        σ_i     = Float64(exp(pf.log_σ[i]))
        mean_r  = (μ - 0.5 * σ_i^2) * dt
        std_r   = σ_i * sqrt(dt)
        pf.log_w[i] += logpdf(Normal(mean_r, std_r), r_t)
    end

    # Guard against complete weight collapse (all -Inf)
    if !any(isfinite, pf.log_w)
        fill!(pf.log_w, 0.0)
    end

    if get_ess(pf) < pf.n / 2
        _systematic_resample!(pf)
        _jitter!(pf)
    end
end

# ---- Resampling internals ----------------------------------------

function _systematic_resample!(pf::ParticleFilter)
    w = get_weights(pf)
    cumw = cumsum(w)
    new_log_σ = similar(pf.log_σ)
    u = rand(Float64) / pf.n
    j = 1
    for i in 1:pf.n
        target = u + (i - 1) / pf.n
        while j < pf.n && cumw[j] < target
            j += 1
        end
        new_log_σ[i] = pf.log_σ[j]
    end
    pf.log_σ .= new_log_σ
    fill!(pf.log_w, 0.0)
end

# Post-resample jitter: small Gaussian noise in log-σ space prevents collapse.
function _jitter!(pf::ParticleFilter; jitter_std::Float32 = 0.02f0)
    pf.log_σ .+= jitter_std .* randn(Float32, pf.n)
    clamp!(pf.log_σ, log(Float32(0.02)), log(Float32(2.0)))
end
