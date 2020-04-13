"""
Reference:
https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html
"""
module eight_schools_centered

using Distributions
using DistributionsAD

schools_dat = Dict(
    :J => 8,
    :y => [28,  8, -3,  7, -1,  1, 18, 12],
    :sigma => [15, 10, 16, 11,  9, 11, 10, 18]
)

# centered 8 schools
function likeli(p)
    mu = p[1]
    tau = exp(p[2])
    theta = p[3:10]
        
    target = 0.0
    
    target += logpdf(Normal(0, 5), mu)
    target += logpdf(Cauchy(0, 5), tau) + p[2]
    target += logpdf.(Normal(mu, tau), theta) |> sum
    target += logpdf.(Normal.(theta, schools_dat[:sigma]), schools_dat[:y]) |> sum
    
    return target
end

end