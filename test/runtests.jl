using Test
@time using YMCMC
import YMCMC: HMC, HMC_DualAveraging, NUTS, NUTS_Naive, mcmc_summary
import YMCMC: Examples

function test_reference_mean(infer::Function, modu::Module; 
        iter=2000, warmup=1000, chains=4, kwargs...)
    iter_sampling = iter - warmup

    size_p = getproperty(modu, :size_p)
    likeli = getproperty(modu, :likeli)
    decode = getproperty(modu, :decode)
    reference_mean = getproperty(modu, :reference_mean)

    posterior = Array{Float64, 3}(undef, chains, iter_sampling, size_p)
    for chain in 1:chains
        fit = infer(likeli, ones(size_p), iter; kwargs...)
        posterior[chain, :, :] = fit[:posterior][end-iter_sampling+1:end, :]
    end

    decoded = decode(posterior)
    df = mcmc_summary(decoded)
    println(df)

    converaged = all(abs.(df[!, :mean] .- reference_mean) .< (4*df[!, :se_mean]))
    if !converaged
        println("Converaged check failed:")
        println([df[!, :mean]; reference_mean; df[!, :se_mean]])
    end
    return converaged
end



@testset "YMCMC" begin
    @testset "eight_schools_non_centered" begin
        # Test every inference method using non-centered eight schools model
        @test @time test_reference_mean(HMC, Examples.eight_schools_non_centered; eps=1e-1, L=15)
        @test @time test_reference_mean(HMC_DualAveraging, Examples.eight_schools_non_centered; delta=0.65, lambda=15., M_adapt=1000)
        @test @time test_reference_mean(NUTS, Examples.eight_schools_non_centered; eps=1e-1)
        @test @time test_reference_mean(NUTS_Naive, Examples.eight_schools_non_centered; eps=1e-1)

        # test other example models
        @test @time test_reference_mean(HMC, Examples.eight_schools_centered; eps=1e-1, L=15)
        @test @time test_reference_mean(HMC, Examples.normal_sample; eps=1e-1, L=15)
    end
end