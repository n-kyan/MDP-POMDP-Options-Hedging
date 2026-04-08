struct OptionContract
    K::Int
    call::Bool
end

struct HedgingSate
    S::Float64
    τ::Float64
    q::Vector or Tuple{Int} # current inventory with length 2, 1 for calls, 1 for puts
    q_spot::Int # current quantity of spot assets
    cash::Float64
    regime_belief::Vector{Float64} # Vector of beliefs of what is current regime
end

struct EnvironmentState
    market_belief::Vector{Float64} # market's returns-only Hamilton filter state
end

struct MarketMakingAction # needs types
    spread_level
    hedge_target
end

struct SimConfig
    S0::Float64 = 100
    K::Int = 100
    r::Float64 = 0.05 # risk free rate
    κ::Float64 = 0.01 # i think this represents transaction cost coefficient - may need tuning
    spread_levels::Vector{Float64} = []
    hedge_targets::Vector{Float64} = []
    A::Float64 # ? what is this constant 
    k::Float64 # ? what is this constant 
    k::Float64 # ? what is this constant 
    Δt::Float64 # ? what is this constant 
    Q::Int # max inventory 
    n_options_per_episode::Int = 8 # number of options lives from start to expiry
end