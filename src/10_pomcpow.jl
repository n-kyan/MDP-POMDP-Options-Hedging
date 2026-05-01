include("7_benchmarks.jl")

using POMDPs
using POMCPOW
using POMDPTools
using BasicPOMCP
using Printf
using ParticleFilters
using Distributions: Normal, pdf
using Statistics: mean, std

# ============================================================
# Section 1: State, observation, and POMDP definition
# ============================================================

# Agent's POMDP state. Contains σ_particle — the agent's continuous vol estimate.
# No regime_idx: the agent does not know the DGP structure.
struct POMDPState
    S::Float64
    σ_particle::Float64
    q_options::Int
    opt_K::Float64
    q_spot::Float64
    cash::Float64
    τ::Float64
    options_completed::Int
end

# POMDP model. ξ is the log-vol diffusion std (annualized).
# The agent uses a log-vol random walk — not Hardy regime switching.
struct OptionsMM_POMDP <: POMDP{POMDPState, MarketMakingAction, OptionsMMObs}
    config::SimConfig
    ξ::Float64   # log-vol diffusion: log(σ_{t+1}) = log(σ_t) + N(0, ξ²·Δt)
end

OptionsMM_POMDP(config::SimConfig) = OptionsMM_POMDP(config, 0.20)

# ============================================================
# Section 2: Initial state distribution
# ============================================================

struct POMDPInitialDist
    config::SimConfig
    μ_log_σ::Float64   # prior mean in log-space
    σ_log_σ::Float64   # prior std in log-space
end

POMDPInitialDist(config::SimConfig) = POMDPInitialDist(config, log(0.20), 0.5)

function Base.rand(rng::AbstractRNG, d::POMDPInitialDist)
    log_σ = d.μ_log_σ + d.σ_log_σ * randn(rng)
    σ     = clamp(exp(log_σ), 0.02, 2.0)
    config = d.config
    return POMDPState(
        config.S0,
        σ,
        0,
        Float64(round(config.S0)),
        0.0,
        0.0,
        config.T_option * config.Δt,
        0,
    )
end

# ============================================================
# Section 3: POMDPs.jl interface
# ============================================================

POMDPs.discount(m::OptionsMM_POMDP) = 0.9999

POMDPs.isterminal(m::OptionsMM_POMDP, s::POMDPState) =
    s.options_completed >= m.config.n_options_per_episode

POMDPs.initialstate(m::OptionsMM_POMDP) = POMDPInitialDist(m.config)

# Generative model: agent's approximate world model.
# Uses log-vol random walk for σ propagation — NO regime_idx, NO Hardy knowledge.
function POMDPs.gen(m::OptionsMM_POMDP, s::POMDPState, a::MarketMakingAction, rng::AbstractRNG)
    config = m.config
    r      = config.r
    Δt     = config.Δt
    σ      = s.σ_particle

    # Agent prices using their vol estimate (hat_V = V_market in agent's model)
    bs    = bs_all(s.S, s.opt_K, s.τ, σ, r; call = true)
    hat_V = bs.price

    # Quotes and fills — symmetric in agent's model since hat_V = V_market_agent
    bid_price = hat_V - a.δ
    ask_price = hat_V + a.δ
    fill      = simulate_fills(bid_price, ask_price, hat_V, config, rng)

    q_new    = s.q_options + (fill.bid_filled ? 1 : 0) - (fill.ask_filled ? 1 : 0)
    cash_new = s.cash
    if fill.bid_filled; cash_new -= bid_price; end
    if fill.ask_filled; cash_new += ask_price; end

    # Hedge toward Δ_target
    hat_Δ_P_post = q_new * bs.Δ + s.q_spot
    shares       = a.Δ_target - hat_Δ_P_post
    hedge_cost   = config.κ * abs(shares) * s.S
    cash_new    -= shares * s.S + hedge_cost
    q_spot_new   = s.q_spot + shares

    # Wealth before (agent's model: option value = hat_V)
    wealth_before = s.cash + s.q_options * hat_V + s.q_spot * s.S

    # Spot step: GBM under agent's σ
    Z     = randn(rng)
    S_new = s.S * exp((r - 0.5σ^2) * Δt + σ * sqrt(Δt) * Z)
    r_t   = log(S_new / s.S)
    τ_new = s.τ - Δt

    # σ propagation: log-vol random walk (agent's model of vol dynamics)
    η         = m.ξ * sqrt(Δt) * randn(rng)
    σ_new     = clamp(exp(log(σ) + η), 0.02, 2.0)

    # Option expiry: settle at contractual payoff — vol-free
    q_opt_new  = q_new
    opt_K_new  = s.opt_K
    opts_done  = s.options_completed
    if τ_new < Δt / 2
        cash_new  += q_opt_new * max(S_new - s.opt_K, 0.0)
        q_opt_new  = 0
        opt_K_new  = Float64(round(S_new))
        opts_done += 1
        τ_new      = config.T_option * Δt
    end

    # Wealth after (agent's model: new BS price at σ_new, S_new)
    bs_new       = q_opt_new > 0 ? bs_all(S_new, opt_K_new, τ_new, σ_new, r; call = true) : nothing
    V_new_agent  = q_opt_new > 0 ? bs_new.price : 0.0
    wealth_after = cash_new + q_opt_new * V_new_agent + q_spot_new * S_new

    reward = (wealth_after - wealth_before) - config.φ * a.Δ_target^2

    sp  = POMDPState(S_new, σ_new, q_opt_new, opt_K_new, q_spot_new, cash_new, τ_new, opts_done)
    obs = OptionsMMObs(r_t, fill.f_t)

    return (sp = sp, o = obs, r = reward)
end

# ============================================================
# Section 4: Observation widening likelihood
# ============================================================

# Weights an observation by its likelihood under the agent's model.
# Primary signal: GBM log-return likelihood given sp.σ_particle.
function POMCPOW.obs_weight(
    m::OptionsMM_POMDP,
    s::POMDPState,
    a::MarketMakingAction,
    sp::POMDPState,
    o::OptionsMMObs,
)
    σ      = sp.σ_particle
    mean_r = (m.config.r - 0.5σ^2) * m.config.Δt
    std_r  = σ * sqrt(m.config.Δt)
    return max(pdf(Normal(mean_r, std_r), o.r_t), 1e-300)
end

# ============================================================
# Section 5: Action widening
# ============================================================

# Extract a representative (S, K, τ, q, q_spot, σ_hat) from any belief type.
# At the root, b is WeightedParticleBelief → use weighted mean σ.
# At tree nodes, b is StateBelief{POWNodeBelief} → rand() samples a state.
function _belief_state_summary(b, config)
    try
        ps    = particles(b)
        ws    = weights(b)
        σ_hat = sum(ws[i] * ps[i].σ_particle for i in eachindex(ps))
        s     = ps[1]
        return s.S, s.opt_K, s.τ, s.q_options, s.q_spot, σ_hat
    catch
        s = rand(b)
        return s.S, s.opt_K, s.τ, s.q_options, s.q_spot, s.σ_particle
    end
end

# Action sampler object — passed as `next_action` to POMCPOWSolver.
# POMCPOW calls: next_action(sampler, pomdp, belief, h_node)
struct POWActionSampler
    config::SimConfig
end

function POMCPOW.next_action(sampler::POWActionSampler, m::OptionsMM_POMDP, b, h)
    config       = sampler.config
    S, K, τ, q, q_spot, σ_hat = _belief_state_summary(b, config)

    bs      = bs_all(S, K, τ, σ_hat, config.r; call = true)
    hat_Δ_P = q * bs.Δ + q_spot

    δ_glft         = glft_half_spread(bs.Γ, S, σ_hat, τ, config)
    δ_glft_clamped = clamp_δ(δ_glft, S, bs.Δ, bs.price, config)

    δ_raw = δ_glft_clamped * exp(0.3 * randn())
    δ     = clamp_δ(δ_raw, S, bs.Δ, bs.price, config)

    Δ_lo     = min(0.0, hat_Δ_P)
    Δ_hi     = max(0.0, hat_Δ_P)
    Δ_target = Δ_lo + rand() * (Δ_hi - Δ_lo + 1e-10)

    return MarketMakingAction(δ, Δ_target)
end

# ============================================================
# Section 6: Rollout policies
# ============================================================

# --- State-based rollout (used inside the POMCPOW tree via FORollout) ---
# Each tree particle has its own σ_particle; the rollout uses that particle's σ.
# FORollout avoids the extract_belief(BootstrapFilter, POWTreeObsNode) issue.
struct GLFTStateRollout <: Policy
    config::SimConfig
end

function POMDPs.action(p::GLFTStateRollout, s::POMDPState)
    config  = p.config
    σ       = s.σ_particle
    bs      = bs_all(s.S, s.opt_K, s.τ, σ, config.r; call = true)
    hat_Δ_P = s.q_options * bs.Δ + s.q_spot
    hat_Γ_P = s.q_options * bs.Γ

    δ_raw    = glft_half_spread(hat_Γ_P, s.S, σ, s.τ, config)
    δ        = clamp_δ(δ_raw, s.S, bs.Δ, bs.price, config)

    H        = ww_band_halfwidth(hat_Γ_P, s.S, σ, config)
    Δ_target = if abs(hat_Δ_P) <= H
        hat_Δ_P
    else
        band_edge = sign(hat_Δ_P) * H
        clamp(band_edge, min(0.0, hat_Δ_P), max(0.0, hat_Δ_P))
    end
    return MarketMakingAction(δ, Δ_target)
end

# --- Belief-based rollout (used as default_action in the outer evaluation loop) ---
# Handles both WeightedParticleBelief (root) and StateBelief (tree nodes).
struct GLFTBeliefRollout <: Policy
    pomdp::OptionsMM_POMDP
end

function POMDPs.action(p::GLFTBeliefRollout, b)
    config           = p.pomdp.config
    S, K, τ, q, q_spot, σ_hat = _belief_state_summary(b, config)

    bs      = bs_all(S, K, τ, σ_hat, config.r; call = true)
    hat_Δ_P = q * bs.Δ + q_spot
    hat_Γ_P = q * bs.Γ

    δ_raw    = glft_half_spread(hat_Γ_P, S, σ_hat, τ, config)
    δ        = clamp_δ(δ_raw, S, bs.Δ, bs.price, config)

    H        = ww_band_halfwidth(hat_Γ_P, S, σ_hat, config)
    Δ_target = if abs(hat_Δ_P) <= H
        hat_Δ_P
    else
        band_edge = sign(hat_Δ_P) * H
        clamp(band_edge, min(0.0, hat_Δ_P), max(0.0, hat_Δ_P))
    end
    return MarketMakingAction(δ, Δ_target)
end

# ============================================================
# Section 7: Solver factory
# ============================================================

function make_pomcpow_solver(pomdp::OptionsMM_POMDP;
                              n_queries::Int    = 50,
                              max_depth::Int    = 5,
                              k_action::Float64 = 4.0,
                              alpha_action::Float64 = 0.5,
                              k_obs::Float64    = 4.0,
                              alpha_obs::Float64 = 0.5,
                              seed::Int         = 42)
    # FORollout: treats rollout as MDP (policy takes state, not belief).
    # Avoids extract_belief(BootstrapFilter, POWTreeObsNode) which is undefined.
    return POMCPOWSolver(
        tree_queries       = n_queries,
        max_depth          = max_depth,
        k_action           = k_action,
        alpha_action       = alpha_action,
        k_observation      = k_obs,
        alpha_observation  = alpha_obs,
        next_action        = POWActionSampler(pomdp.config),
        estimate_value     = BasicPOMCP.FORollout(GLFTStateRollout(pomdp.config)),
        default_action     = GLFTBeliefRollout(pomdp),
        rng                = MersenneTwister(seed),
        enable_action_pw   = true,
    )
end

# ============================================================
# Section 8: Belief utilities
# ============================================================

# Extract σ_hat from a POMCPOW particle belief.
# Normalizes weights because BootstrapFilter can return unnormalized likelihoods
# when ESS is high enough that resampling does not trigger.
function belief_mean_σ(b)::Float64
    ps    = particles(b)
    ws    = weights(b)
    total = sum(ws)
    return sum(ws[i] * ps[i].σ_particle for i in eachindex(ps)) / total
end

# Build an initial WeightedParticleBelief from the prior distribution.
function make_initial_belief(m::OptionsMM_POMDP, rng::AbstractRNG, n::Int = 500)
    d  = POMDPInitialDist(m.config)
    ps = [rand(rng, d) for _ in 1:n]
    ws = fill(1.0 / n, n)
    return WeightedParticleBelief(ps, ws)
end

# ============================================================
# Section 9: Evaluation
# ============================================================

function evaluate_pomcpow(
    vm::VolModel,
    config::SimConfig,
    n_episodes::Int,
    seed::Int;
    n_queries::Int   = 50,
    max_depth::Int   = 5,
    ξ::Float64       = 0.20,
    n_particles::Int = 500,
)
    pomdp    = OptionsMM_POMDP(config, ξ)
    solver   = make_pomcpow_solver(pomdp; n_queries, max_depth, seed)
    planner  = solve(solver, pomdp)
    up       = updater(planner)

    rng_env  = MersenneTwister(seed + 1)
    pf_dummy = ParticleFilter(10)

    episode_rewards = Float64[]

    for ep in 1:n_episodes
        # Initialize TRUE Hardy environment
        env = EnvironmentState(
            AgentState(NaN, 0.0, 0.0, 0, config.T_option * config.Δt, 0.2, 0.0, 0.0, config.S0),
            VolState(vm),
            OptionContract[OptionContract(Float64(round(config.S0)), true)],
            0,
        )
        portfolio = Portfolio()
        push!(portfolio.option_quantities, 0)
        initialize_episode!(env, portfolio, vm, config, pf_dummy)

        # Initialize POMCPOW belief from log-normal prior
        belief = make_initial_belief(pomdp, rng_env, n_particles)

        ep_reward = 0.0
        done      = false

        while !done
            # σ_hat from POMCPOW belief (not from pf_dummy)
            σ_hat  = belief_mean_σ(belief)
            action = POMDPs.action(planner, belief)

            # Execute in TRUE environment; σ_hat_override routes POMCPOW belief to quotes
            _, reward, done, info = step_environment!(
                env, portfolio, pf_dummy, action, config, rng_env;
                σ_hat_override = σ_hat,
            )
            ep_reward += reward

            # Update POMCPOW belief with true observation (agent's model used for propagation)
            true_obs = OptionsMMObs(info.log_return, info.fill.f_t)
            belief   = POMDPs.update(up, belief, action, true_obs)

            # Rao-Blackwellization: override fully-observable components with true values.
            # Only σ_particle remains uncertain; everything else the agent can observe directly.
            S_true    = env.agent_state.S
            q_true    = portfolio.option_quantities[1]
            qs_true   = portfolio.q_spot
            cash_true = portfolio.cash
            τ_true    = env.agent_state.τ
            K_true    = isempty(env.current_options) ? Float64(round(S_true)) : env.current_options[1].K
            done_true = env.options_completed
            ps_synced = [POMDPState(S_true, p.σ_particle, q_true, K_true, qs_true, cash_true, τ_true, done_true)
                         for p in particles(belief)]
            ws_raw    = weights(belief)
            belief    = WeightedParticleBelief(ps_synced, ws_raw ./ sum(ws_raw))
        end

        push!(episode_rewards, ep_reward)
        ep % 10 == 0 && @printf("  Episode %d/%d done\n", ep, n_episodes)
    end

    μ      = mean(episode_rewards)
    σ_pnl  = std(episode_rewards)
    sharpe = σ_pnl > 1e-10 ? μ / σ_pnl : 0.0

    return (episode_rewards = episode_rewards, mean_reward = μ, std_reward = σ_pnl, sharpe = sharpe)
end

# ============================================================
# Section 10: Diagnostic trace
# ============================================================

# Per-step comparison: POMCPOW δ vs GLF-T δ at the same σ_hat.
# Prints: step | true_σ | σ_hat | δ_pomcpow | δ_glft | fill_dir | q_after
function diagnose_pomcpow(
    vm::VolModel,
    config::SimConfig;
    n_episodes::Int  = 5,
    seed::Int        = 42,
    n_queries::Int   = 50,
    max_depth::Int   = 5,
    ξ::Float64       = 0.20,
    n_particles::Int = 500,
)
    pomdp   = OptionsMM_POMDP(config, ξ)
    solver  = make_pomcpow_solver(pomdp; n_queries, max_depth, seed)
    planner = solve(solver, pomdp)
    up      = updater(planner)
    rng_env = MersenneTwister(seed + 1)
    pf_dummy = ParticleFilter(10)

    @printf("%-4s  %-7s  %-7s  %-10s  %-10s  %-5s  %-7s\n",
            "step", "true_σ", "σ_hat", "δ_pomcpow", "δ_glft", "fill", "q_after")
    println(repeat("-", 60))

    for ep in 1:n_episodes
        env = EnvironmentState(
            AgentState(NaN, 0.0, 0.0, 0, config.T_option * config.Δt, 0.2, 0.0, 0.0, config.S0),
            VolState(vm),
            OptionContract[OptionContract(Float64(round(config.S0)), true)],
            0,
        )
        portfolio = Portfolio()
        push!(portfolio.option_quantities, 0)
        initialize_episode!(env, portfolio, vm, config, pf_dummy)
        belief = make_initial_belief(pomdp, rng_env, n_particles)

        step_deltas   = Float64[]  # δ_pomcpow
        step_glft     = Float64[]  # δ_glft at same σ_hat
        n_narrower    = 0
        ep_reward     = 0.0
        done          = false
        t             = 0

        println("\n=== Episode $ep ===")
        while !done
            t += 1
            σ_hat  = belief_mean_σ(belief)
            action = POMDPs.action(planner, belief)

            # true σ from regime transition row
            ws_regime = perfect_regime_belief(env.vol_state)
            true_σ    = sqrt(sum(ws_regime .* vm.σ_levels .^ 2))

            # GLF-T δ at same σ_hat and current portfolio Greeks
            hat_Γ_P = env.agent_state.hat_Γ_P
            bs_glft = bs_all(env.agent_state.S, env.current_options[1].K,
                              env.agent_state.τ, σ_hat, config.r; call = true)
            δ_glft_raw = glft_half_spread(hat_Γ_P, env.agent_state.S, σ_hat, env.agent_state.τ, config)
            δ_glft     = clamp_δ(δ_glft_raw, env.agent_state.S, bs_glft.Δ, bs_glft.price, config)

            _, reward, done, info = step_environment!(
                env, portfolio, pf_dummy, action, config, rng_env;
                σ_hat_override = σ_hat,
            )
            ep_reward += reward

            q_after   = portfolio.option_quantities[1]
            fill_dir  = info.fill.f_t

            @printf("%-4d  %-7.4f  %-7.4f  %-10.4f  %-10.4f  %-5d  %-7d\n",
                    t, true_σ, σ_hat, action.δ, δ_glft, fill_dir, q_after)

            push!(step_deltas, action.δ)
            push!(step_glft,   δ_glft)
            action.δ < δ_glft && (n_narrower += 1)

            true_obs = OptionsMMObs(info.log_return, info.fill.f_t)
            belief   = POMDPs.update(up, belief, action, true_obs)
            ws_raw   = weights(belief)
            ps_sync  = [POMDPState(env.agent_state.S, p.σ_particle,
                                   portfolio.option_quantities[1],
                                   isempty(env.current_options) ? Float64(round(env.agent_state.S)) : env.current_options[1].K,
                                   portfolio.q_spot, portfolio.cash,
                                   env.agent_state.τ, env.options_completed)
                        for p in particles(belief)]
            belief = WeightedParticleBelief(ps_sync, ws_raw ./ sum(ws_raw))
        end

        mean_δ_ratio = mean(step_deltas ./ step_glft)
        pct_narrower = 100 * n_narrower / length(step_deltas)
        @printf("  → ep_reward=%.2f  mean_δ_ratio=%.3f  pct_narrower=%.1f%%\n",
                ep_reward, mean_δ_ratio, pct_narrower)
    end
end
