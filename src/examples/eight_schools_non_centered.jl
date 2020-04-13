"""
Reference:
https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html
"""
module eight_schools_non_centered

using Distributions
using DistributionsAD

schools_dat = Dict(
    :J => 8,
    :y => [28,  8, -3,  7, -1,  1, 18, 12],
    :sigma => [15, 10, 16, 11,  9, 11, 10, 18]
)

function likeli(p)
    mu = p[1]
    tau = exp(p[2])
    theta_tilde = p[3:10]
    
    theta = mu .+ tau * theta_tilde
    
    target = 0
    target += logpdf(Normal(0, 5), mu)
    target += logpdf(Cauchy(0, 5), tau) + p[2]
    target += logpdf.(Normal(0, 1), theta_tilde) |> sum
    target += logpdf.(Normal.(theta, schools_dat[:sigma]), schools_dat[:y]) |> sum
    
    return target
end

function recover_lp(mu, theta_t, tau)
    
    theta = mu .+ theta_t * tau
    
    target = 0
    target += logpdf(Normal(0, 5), mu)
    target += logpdf(Cauchy(0, 5), tau) + log(tau) + log(2)
    target += logpdf.(Normal(0, 1), theta_t) |> sum
    target += logpdf.(Normal.(theta, schools_dat[:sigma]), schools_dat[:y]) |> sum
    
    return target
end

end