struct DiscretizationScheme
    Δ_edges::Vector{Float64}
    Γ_edges::Vector{Float64}
    time_edges::Vector{Int}
    hedge_ratio_edges::Vector{Float64}
end

function default_scheme(T::Int)
    Δ_edges = collect(range(-0.05, 1.05, length=12))
    Γ_edges = [0.0, 0.005, 0.015, 0.03, 0.055, 0.09, Inf]
    time_step   = max(1, T ÷ 8)
    time_edges  = unique(vcat(collect(0:time_step:T), [T, T+1]))
    hedge_ratio_edges = [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    return DiscretizationScheme(Δ_edges, Γ_edges, time_edges, hedge_ratio_edges)
end

function bin_index(val, edges)
    n = length(edges) - 1
    for i in 1:n
        if val < edges[i+1]
            return i
        end
    end
    return n
end

function hedge_ratio(state::StepState, params::EnvironmentParams)
    Δ, _ = portfolio_greeks(params.port, state.S, params.r, params.σ, state.t, params.dt)
    abs(Δ) < 1e-6 && return 0.0
    return state.hedge_position / Δ
end

function discretize(state::StepState, params::EnvironmentParams, scheme::DiscretizationScheme)
    Δ, Γ = portfolio_greeks(params.port, state.S, params.r, params.σ, state.t, params.dt)
    h     = hedge_ratio(state, params)
    i_Δ   = bin_index(Δ, scheme.Δ_edges)
    i_Γ   = bin_index(Γ, scheme.Γ_edges)
    i_t   = bin_index(state.t, scheme.time_edges)
    i_h   = bin_index(h, scheme.hedge_ratio_edges)
    return i_Δ, i_Γ, i_t, i_h
end

function state_index(i_Δ::Int, i_Γ::Int, i_t::Int, i_h::Int, scheme::DiscretizationScheme)
    n_Δ = length(scheme.Δ_edges) - 1
    n_Γ = length(scheme.Γ_edges) - 1
    n_t = length(scheme.time_edges) - 1
    return (i_h - 1) * n_Δ * n_Γ * n_t +
           (i_t - 1) * n_Δ * n_Γ +
           (i_Γ - 1) * n_Δ +
           i_Δ
end

function n_states(scheme::DiscretizationScheme)
    return (length(scheme.Δ_edges) - 1) *
           (length(scheme.Γ_edges) - 1) *
           (length(scheme.time_edges) - 1) *
           (length(scheme.hedge_ratio_edges) - 1)
end

function build_representative_states(
    env_params::EnvironmentParams,
    scheme::DiscretizationScheme;
    n_paths::Int = 2000,
    rng::AbstractRNG = MersenneTwister(0)
)
    T = env_params.port.positions[1].T

    S_sums     = Dict{NTuple{4,Int}, Float64}()
    hedge_sums = Dict{NTuple{4,Int}, Float64}()
    counts     = Dict{NTuple{4,Int}, Int}()

    prices_mat, _ = simulate_gbm(
        100.0, env_params.r, env_params.σ, env_params.dt, T;
        n_paths=n_paths, rng=rng
    )

    for p in 1:n_paths
        hedge = 0.0
        for step in 0:T-1
            S     = prices_mat[step+1, p]
            state = StepState(S, step, hedge)
            key   = discretize(state, env_params, scheme)

            S_sums[key]     = get(S_sums,     key, 0.0) + S
            hedge_sums[key] = get(hedge_sums, key, 0.0) + hedge
            counts[key]     = get(counts,     key, 0)   + 1

            Δ, _ = portfolio_greeks(env_params.port, S, env_params.r,
                                    env_params.σ, step, env_params.dt)
            hedge = Δ * 0.5
        end
    end

    rep_states = Dict{NTuple{4,Int}, StepState}()
    for (key, count) in counts
        S_mean    = S_sums[key]    / count
        h_mean    = hedge_sums[key] / count
        i_Δ, i_Γ, i_t, i_h = key
        t_rep     = scheme.time_edges[i_t]
        rep_states[key] = StepState(S_mean, t_rep, h_mean)
    end

    return rep_states, counts
end