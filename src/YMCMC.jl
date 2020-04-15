module YMCMC

greet() = print("Hello World!")

using Distributions
using DistributionsAD
using ForwardDiff
using OffsetArrays

using DataFrames
import StatsBase: autocor

include("HMC.jl")
include("NUTS.jl")
include("diagnosis.jl")
include("examples/Examples.jl")

end # module
