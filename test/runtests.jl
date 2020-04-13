using Test
import YMCMC: HMC, HMC_DualAveraging, mcmc_summary
import YMCMC.examples: eight_schools_non_centered

@testset "YMCMC" begin
    @testset "eight_schools_non_centered" begin
        iter = 1000
        chains = 4
        warmup = iter ÷ 2
        iter_sampling = iter - warmup
        
        posterior = Array{Float64, 3}(undef, chains, iter_sampling, 10)
        for chain in 1:chains
            fit = HMC(ones(10), 1e-1, 30, eight_schools_non_centered.likeli, iter) # (draws, parameters)
            posterior[chain, :, :] = fit[:posterior][warmup+1:end, :]
        end

        decoded = copy(posterior)
        decoded[:, :, 2] = exp.(decoded[:, :, 2])
        decoded[:, :, 3:end] = decoded[:, :, 3:end] .* decoded[:, :, 2] .+ decoded[:, :, 1]

        df = mcmc_summary(decoded)
        
        reference_mean = [4.41, 3.51, 6.2, 4.81, 3.97, 4.76, 3.63, 4.08, 6.22, 4.82]
        controlled = abs.(df[!, :mean] .- reference_mean) .< (5*df[!, :se_mean])
        @test all(controlled)


        iter = 1000
        chains = 4
        warmup = iter ÷ 2
        iter_sampling = iter - warmup
        
        posterior = Array{Float64, 3}(undef, chains, iter_sampling, 10)
        for chain in 1:chains
            fit = HMC_DualAveraging(ones(10), 0.65, 15, eight_schools_non_centered.likeli, iter, warmup)
            posterior[chain, :, :] = fit[:posterior][warmup+2:end, :]
        end

        decoded = copy(posterior)
        decoded[:, :, 2] = exp.(decoded[:, :, 2])
        decoded[:, :, 3:end] = decoded[:, :, 3:end] .* decoded[:, :, 2] .+ decoded[:, :, 1]
        
        df = mcmc_summary(decoded)

        reference_mean = [4.41, 3.51, 6.2, 4.81, 3.97, 4.76, 3.63, 4.08, 6.22, 4.82]
        controlled = abs.(df[!, :mean] .- reference_mean) .< (5*df[!, :se_mean])
        @test all(controlled)
    end
end