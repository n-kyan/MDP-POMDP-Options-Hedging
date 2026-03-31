# sandbox.jl

includet("src/simulation.jl")
includet("src/types.jl")
includet("src/greeks.jl")
includet("src/environment.jl")
includet("src/discretization.jl")
includet("src/value_iteration.jl")

verify_value_iteration()

csv_text = read("results/sensitivity_grid.csv", String)