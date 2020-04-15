"""
    NUTS(
        theta0::AbstractVector{<:Real}, 
        eps::Real,
        likeli::Function, 
        M::Int,
        delta_max::Real
    )

Naive_NUTS, see: The No-U-Turn Sampler: Adaptively Setting Path Lengths 
in Hamiltonian Monte Carlo

# Arguments
- `theta0`: initial value
- `eps`: step size
- `likeli`: joint probability of model
- `M`: size of draws
- `delta_max`: 
"""
function NUTS(
        likeli::Function,
        theta0::Vector{T}, 
        M::Int;
        eps::T,

        delta_max::T = 1000.,
        max_tree_depth::Int = 15,
    ) where T <: Real
    size_p = size(theta0, 1)
    theta_arr = Array{Float64, 2}(undef, M+1, size_p)
    theta_arr[1, :] = theta0
    for m = 2:M+1
        r0 = rand(Normal(), size_p)
        # u = rand() * exp(likeli(theta0) - 0.5 * r0' * r0) # slice
        # Stan's trick: 
        # https://github.com/stan-dev/stan/blob/736311d88e99b997f5b902409752fb29d6ec0def/src/stan/mcmc/hmc/nuts_classic/base_nuts_classic.hpp#L197
        # X ~ U(0, a) Z ~ U(0,1), log(X) = log(aZ) = log(Z) + log(a)
        log_u = log(rand()) + likeli(theta_arr[m-1, :]) - 0.5 * r0' * r0
        theta_neg = theta_pos = theta_arr[m-1, :]
        r_neg = r_pos = r0
        j = 0
        theta_arr[m, :] = theta_arr[m-1, :]
        n = 1
        s = true
        
        while s
            if j > max_tree_depth
                println("Reach max-depth tree, please increase eps or max_tree_depth")
                break
            end
            
            v = rand() < 0.5 ? -1 : 1
            if v == -1
                theta_neg, r_neg, _, _, theta_p, n_p, s_p = BuildTree(likeli, theta_neg, r_neg, log_u, v, j, eps, delta_max)
            else
                _, _, theta_pos, r_pos, theta_p, n_p, s_p = BuildTree(likeli, theta_pos, r_pos, log_u, v, j, eps, delta_max) 
            end
            
            if s_p
                #if rand() < (n_p / n)
                if rand() < (n_p / (n + n_p))
                    theta_arr[m, :] = theta_p
                end
            end
            
            n = n + n_p
            theta_diff = theta_pos .- theta_neg
            s = s_p & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
            j = j + 1
        end

        #println("m: $m, j: $j, n: $n, ")
    end

    return Dict(:posterior => collect(theta_arr[1:end, :]))
end

"""
    BuildTree(
        theta::AbstractVector{<:Real}, 
        r::AbstractVector{<:Real},
        u::Real,
        v::Int, 
        j::Int,
        eps::Real,
        likeli::Function,
        delta_max::Real
    )

# Arguments

- `theta`: parameters
- `r`: momentum
- `u`: slice
- `v`: direction
- `j`: tree depth
- `eps`: step size
- `likeli`: joint probability up to a constant
- `delta_max`: a threshold

# Return

- `theta_neg`: theta on negative direction pointer
- `r_neg`: 
- `theta_pos`:
- `r_pos`:
- `theta_p`: "selected" theta
- `n_p`: weight 
- `s_p`: continue?
"""
function BuildTree(
        likeli::Function,

        theta::Vector{T}, 
        r::Vector{T},
        log_u::T,
        v::Int, 
        j::Int,
        eps::T,

        delta_max::T
    ) where T <: Real
    if j == 0
        # Base case, take one leapfrog step in direction v
        theta_p, r_p = leapfrog(likeli, theta, r, v * eps)
        energy = likeli(theta_p) - 0.5 * r_p' * r_p
        n_p = (log_u <= energy) * 1.0  # x * 1.0 is used as indicator function
        s_p = energy > (log_u - delta_max)
        if !s_p
            println("divergent detected")
        end
        return theta_p, r_p, theta_p, r_p, theta_p, n_p, s_p
    else
        # Recursion - implicitly build the left and right subtree
        theta_neg, r_neg, theta_pos, r_pos, theta_p, n_p, s_p = BuildTree(likeli, theta, r, log_u, v, j-1, eps, delta_max)
        if s_p
            if v == -1
                theta_neg, r_neg, _, _, theta_pp, n_pp, s_pp = BuildTree(likeli, theta_neg, r_neg, log_u, v, j-1, eps, delta_max) 
            else
                _, _, theta_pos, r_pos, theta_pp, n_pp, s_pp = BuildTree(likeli, theta_pos, r_pos, log_u, v, j-1, eps, delta_max)
            end
            if rand() < (n_pp / (n_p + n_pp))
                theta_p = theta_pp
            end
            theta_diff = theta_pos - theta_neg
            s_p = s_pp & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
            n_p = n_p + n_pp
        end
        return theta_neg, r_neg, theta_pos, r_pos, theta_p, n_p, s_p
    end
end


function NUTS_Naive(
        likeli::Function,
        theta0::Vector{T},
        M::Int;

        eps::T, 
        delta_max = 1000.,
        max_tree_depth::Int = 15
    ) where T <: Real
    size_p = length(theta0)
    theta_arr = Array{T, 2}(undef, M+1, size_p)
    theta_arr[1, :] = theta0

    for m = 2:M+1
        r0 = rand(Normal(), size_p)
        #u = rand() * exp(likeli(theta_arr[m-1, :]) - 0.5 * r0' * r0) # slice
        log_u = log(rand()) + likeli(theta_arr[m-1, :]) - 0.5 * r0' * r0
        theta_neg = theta_pos = theta_arr[m-1, :]
        r_neg = r_pos = r0
        j = 0
        C = Tuple{Vector{T}, Vector{T}}[]
        push!(C, (theta_arr[m-1, :], r0))
        s = true

        while s
            if j > max_tree_depth
                println("Reach max-depth tree, please increase eps or max_tree_depth")
                break
            end

            v = rand() < 0.5 ? -1 : 1
            if v == -1
                theta_neg, r_neg, _, _, C_p, s_p = BuildTree_Naive(likeli, theta_neg, r_neg, log_u, v, j, eps, delta_max)
            else
                _, _, theta_pos, r_pos, C_p, s_p = BuildTree_Naive(likeli, theta_pos, r_pos, log_u, v, j, eps, delta_max)
            end

            if s_p
                C = [C; C_p]
            end

            theta_diff = theta_pos .- theta_neg
            s = s_p & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
            j = j + 1
        end

        theta_selected, r_selected = rand(C)
        theta_arr[m, :] = theta_selected
    end

    return Dict{Symbol, Array{T, 2}}(:posterior => theta_arr)
end

function BuildTree_Naive(
        likeli::Function,

        theta::Vector{T},
        r::Vector{T},
        log_u::T,
        v::Int,
        j::Int,
        eps::T,

        delta_max::T
    ) where T <: Real

    if j == 0
        # Base case - take one leapfrog step in the direction v
        theta_p, r_p = leapfrog(likeli, theta, r, v * eps)
        energy = likeli(theta_p) - 0.5 * r_p' * r_p
        C_p = Tuple{Vector{T}, Vector{T}}[]
        if log_u <= energy
            push!(C_p, (theta_p, r_p))
        end
        s_p = energy > (log_u - delta_max)
        if !s_p
            println("divergent detected")
        end
        return theta_p, r_p, theta_p, r_p, C_p, s_p
    else
        # Recursion - build the left and right subtrees
        theta_neg, r_neg, theta_pos, r_pos, C_p, s_p = BuildTree_Naive(likeli, theta, r, log_u, v, j-1, eps, delta_max)
        if v == -1
            theta_neg, r_neg, _, _, C_pp, s_pp = BuildTree_Naive(likeli, theta_neg, r_neg, log_u, v, j-1, eps, delta_max)
        else
            _, _, theta_pos, r_pos, C_pp, s_pp = BuildTree_Naive(likeli, theta_pos, r_pos, log_u, v, j-1, eps, delta_max)
        end

        theta_diff = theta_pos - theta_neg
        s_p = s_p & s_pp & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
        C_p = [C_p ; C_pp]
        return theta_neg, r_neg, theta_pos, r_pos, C_p, s_p
    end
end



