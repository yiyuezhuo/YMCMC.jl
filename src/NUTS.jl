"""
    NUTS(
        likeli::Function,
        theta0::Vector{T}, 
        M::Int;
        eps::T,

        delta_max::T = 1000.,
        max_tree_depth::Int = 15,
    ) where T <: Real

Naive_NUTS, see: The No-U-Turn Sampler: Adaptively Setting Path Lengths 
in Hamiltonian Monte Carlo

# Arguments

- `likeli`: joint probability of model
- `theta0`: initial value
- `M`: size of draws
- `eps`: step size
- `delta_max`: 
"""
function NUTS(
        likeli::Function,
        theta0::Vector{T}, 
        M::Int;
        eps::T,

        delta_max::T = one(T) * 1000,
        max_tree_depth::Int = 15,
    ) where T <: Real
    size_p = size(theta0, 1)
    theta_arr = Array{T, 2}(undef, M+1, size_p)
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
                println("Reach max-depth tree: $j > $max_tree_depth, please increase eps or max_tree_depth")
                break
            end
            
            v = rand() < 0.5 ? -1 : 1
            if v == -1
                theta_neg, r_neg, _, _, theta_p, n_p, s_p = BuildTree(likeli, theta_neg, r_neg, log_u, v, j, eps, delta_max)
            else
                _, _, theta_pos, r_pos, theta_p, n_p, s_p = BuildTree(likeli, theta_pos, r_pos, log_u, v, j, eps, delta_max) 
            end
            
            if s_p
                if rand() < (n_p / n)
                #if rand() < (n_p / (n + n_p))
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
        likeli::Function,

        theta::Vector{T}, 
        r::Vector{T},
        log_u::T,
        v::Int, 
        j::Int,
        eps::T,

        delta_max::T
    ) where T <: Real

# Arguments

- `likeli`: joint probability up to a constant
- `theta`: parameters
- `r`: momentum
- `u`: slice
- `v`: direction
- `j`: tree depth
- `eps`: step size
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
            println("divergent detected: $energy <= $log_u - $delta_max")
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
                println("Reach max-depth tree $j > $max_tree_depth, please increase eps or max_tree_depth")
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
            println("divergent detected $energy <= $log_u - $delta_max")
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


function NUTS_DualAveraging(
    likeli::Function,
    theta0::Vector{T},
    M::Int;

    delta::T,
    M_adapt::Int,
    delta_max::T = one(T) * 1000,
    max_tree_depth::Int = 15
    )where T <: Real

    size_p = length(theta0)

    eps0 = FindReasonableEpsilon(likeli, theta0)
    #println("eps0 $eps0")
    mu = log(10*eps0)
    gamma = 0.05
    t0 = 10
    kappa = 0.75

    theta_arr = Array{T, 2}(undef, M+1, size_p)
    eps_arr = Vector{T}(undef, M+1)
    H_bar_arr = Vector{T}(undef, M_adapt+1)
    eps_bar_arr = Vector{T}(undef, M_adapt+1)

    theta_arr[1, :] = theta0
    eps_arr[1] = eps0
    H_bar_arr[1] = 0
    eps_bar_arr[1] = 1

    for m = 2:M+1
        r0 = randn(size_p)
        # X ~ U(0,1), Y ~ U(0,a), log(Y)=log(aX)=log(a)+log(X)
        energy0 = likeli(theta_arr[m-1, :]) - 0.5 * r0' * r0
        log_u = log(rand()) + energy0
        theta_neg = theta_pos = theta_arr[m-1, :]
        r_neg = r_pos = r0
        j = 0
        theta_arr[m, :] = theta_arr[m-1, :]
        n = 1
        s = true

        alpha = 0.0
        n_alpha = 0.0
        while s
            if j > max_tree_depth
                println("Reach max_tree_depth detected: $j > $max_tree_depth")
                break
            end
            v = rand() < 0.5 ? -1 : 1
            if v == -1
                theta_neg, r_neg, _, _, theta_p, n_p, s_p, alpha, n_alpha = BuildTree_DualAveraging(likeli, theta_neg, r_neg, log_u, v, j, eps_arr[m-1], energy0, delta_max)
            else
                _, _, theta_pos, r_pos, theta_p, n_p, s_p, alpha, n_alpha = BuildTree_DualAveraging(likeli, theta_pos, r_pos, log_u, v, j, eps_arr[m-1], energy0, delta_max)
            end

            if s_p
                #if rand() < n_p / (n+n_p)
                if rand() < n_p / n
                    theta_arr[m, :] = theta_p
                end
            end

            n = n + n_p
            theta_diff = theta_pos .- theta_neg
            s = s_p & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
            j = j+1
        end

        if m <= M_adapt+1
            H_bar_arr[m] = (1 - 1/(m+t0)) * H_bar_arr[m-1] + 1/(m+t0) * (delta - alpha / n_alpha)
            eps_arr[m] = exp(mu - sqrt(m)/gamma * H_bar_arr[m])
            w = m^(-kappa)
            eps_bar_arr[m] = exp(w * log(eps_arr[m]) + (1-w) * log(eps_bar_arr[m-1]))
        else
            eps_arr[m] = eps_bar_arr[M_adapt+1]
        end
    end

    return Dict(:posterior => theta_arr, :eps_arr => eps_arr,
                :H_bar_arr => H_bar_arr, :eps_bar_arr => eps_bar_arr)
end

function BuildTree_DualAveraging(
        likeli::Function,
        theta::Vector{T},
        r::Vector{T},
        log_u::T,
        v::Int,
        j::Int,
        eps::T,
        energy0::T,
        delta_max::T
    ) where T <: Real

    if j == 0
        # Base case - take one leapfrog step in the direction v.
        theta_p, r_p = leapfrog(likeli, theta, r, v*eps)
        energy = likeli(theta_p) - 0.5 * r' * r
        n_p = log_u <= energy
        s_p = log_u <= delta_max + energy
        if !s_p
            println("Divergent detected $log_u > $delta_max + $energy")
        end
        return theta_p, r_p, theta_p, r_p, theta_p, n_p, s_p, min(1, exp(energy - energy0)), 1.
    else
        # Recursion
        theta_neg, r_neg, theta_pos, r_pos, theta_p, n_p, s_p, alpha_p, n_alpha_p = BuildTree_DualAveraging(likeli, theta, r, log_u, v, j-1, eps, energy0, delta_max)
        if s_p
            if v == -1
                theta_neg, r_neg, _, _, theta_pp, n_pp, s_pp, alpha_pp, n_alpha_pp = BuildTree_DualAveraging(likeli, theta_neg, r_neg, log_u, v, j-1, eps, energy0, delta_max)
            else
                _, _, theta_pos, r_pos, theta_pp, n_pp, s_pp, alpha_pp, n_alpha_pp = BuildTree_DualAveraging(likeli, theta_pos, r_pos, log_u, v, j-1, eps, energy0, delta_max)
            end

            if rand() < (n_pp) / (n_p + n_pp)
                theta_p = theta_pp
            end

            alpha_p = alpha_p + alpha_pp
            n_alpha_p = n_alpha_p + n_alpha_pp

            theta_diff = theta_pos .- theta_neg
            s_p = s_pp & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
            n_p = n_p + n_pp
        end
        return theta_neg, r_neg, theta_pos, r_pos, theta_p, n_p, s_p, alpha_p, n_alpha_p
    end
end
