module normal_sample

using Distributions
using DistributionsAD

function likeli(p)
    x, y = p
    
    target = 0
    
    target += logpdf(Normal(0, 1), x)
    target += logpdf(Normal(1, 2), y)
    
    return target
end

reference_mean = [0., 1.]

decode(x) = x
size_p = 2

end