module YMCMC

greet() = print("Hello World!")

using Distributions
using DistributionsAD
using ForwardDiff
using OffsetArrays

using DataFrames
import StatsBase: autocor

include("HMC.jl")
include("diagnosis.jl")
include("examples/examples.jl")

end # module
