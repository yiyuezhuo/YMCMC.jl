"""
Arguments:
    theta0: initial parameters value
    eps: step size
    L: iteration length
    likeli: log lielihood function, Array{Float64, 1} -> Float64
    M: sample size
Return:
    Array: draws x parameters
"""
function HMC(theta0, eps, L, likeli, M)
    n = size(theta0, 1)
    theta_arr = Array{Float64, 2}(undef, M+1, n)
    theta_arr = OffsetArray(theta_arr, 0:M, 1:n)
    theta_arr[0, :] = theta0
    for m = 1:M
        r0 = rand(Normal(), n)
        theta_arr[m, :] = theta_arr[m-1, :]
        theta_tilde = theta_arr[m-1, :]
        r_tilde = r0
        for i = 1:L
            r_tilde = r_tilde + (eps/2) * ForwardDiff.gradient(likeli, theta_tilde)
            theta_tilde = theta_tilde + eps * r_tilde
            r_tilde = r_tilde + (eps/2) * ForwardDiff.gradient(likeli, theta_tilde)
        end
        ratio_upper = exp(likeli(theta_tilde) - 0.5 * (r_tilde' * r_tilde) )
        ratio_bottom = exp(likeli(theta_arr[m-1, :]) - 0.5 * (r0' * r0) )
        alpha = min(1, ratio_upper / ratio_bottom)
        if rand() < alpha
            theta_arr[m, :] = theta_tilde
            # r_m = - r_tilde
        end
    end
    return Dict(:posterior => collect(theta_arr[1:end, :]))
end

function leapfrog(likeli, theta_tilde, r_tilde, eps)
    r_tilde = r_tilde + (eps/2) * ForwardDiff.gradient(likeli, theta_tilde)
    theta_tilde = theta_tilde + eps * r_tilde
    r_tilde = r_tilde + (eps/2) * ForwardDiff.gradient(likeli, theta_tilde)
    return theta_tilde, r_tilde
end

function get_accept(likeli, theta_p, r_p, theta, r)
    logp_theta_p_r_p = likeli(theta_p) - 0.5 * (r_p' * r_p)
    logp_theta_r = likeli(theta) - 0.5 * (r' * r)
    accept_p = min(1., exp(logp_theta_p_r_p - logp_theta_r))
    return accept_p
end

function FindReasonableEpsilon(likeli, theta)
    n = size(theta, 1)
    eps = 1.
    r = rand(Normal(), n)
    theta_p, r_p = leapfrog(likeli, theta, r, eps)
    accept = get_accept(likeli, theta_p, r_p, theta, r)
    # println("init accept: $accept")
    if accept < 0.5
        while accept < 0.5
            eps /= 1.1  # 2
            theta_p, r_p = leapfrog(likeli, theta, r, eps)
            accept = get_accept(likeli, theta_p, r_p, theta, r)
            # println("> accept: $accept")
        end
    else
        while accept > 0.5
            eps *= 1.1  # 2
            theta_p, r_p = leapfrog(likeli, theta, r, eps)
            accept = get_accept(likeli, theta_p, r_p, theta, r)
            # println("< accept: $accept")
        end
    end
    return eps
end

"""
Arguments:
    theta0: initial value
    delta: target accept probability
    lambda: "invariant" step length
    likeli: sampled distribution (maybe up to a constant)
    M: total sample size
    M_adapt: warmup sample size
Return:
    theta_arr: Array, (draws, parameters)
"""
function HMC_DualAveraging(theta0, delta, lambda, likeli, M, M_adapt)
    n = size(theta0, 1)
    
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    
    theta_arr = Array{Float64, 2}(undef, M+1, n)
    theta_arr[1, :] = theta0
    
    eps_arr = Array{Float64, 1}(undef, M+1)
    eps_arr[1] = FindReasonableEpsilon(likeli, theta0)
    
    eps_bar_arr = Array{Float64, 1}(undef, M_adapt+1)
    eps_bar_arr[1] = 1.0
    
    H_arr = Array{Float64, 1}(undef, M_adapt+1)
    H_arr[1] = 0
    
    accept_arr = Array{Float64, 1}(undef, M)
    L_arr = Array{Int, 1}(undef, M)
    
    mu = log(10*eps_arr[1])
    
    for m = 2:M+1
        #println("m: $m, eps_arr[m-1]: $(eps_arr[m-1])")
        r0 = rand(Normal(), n)
        theta_arr[m, :] = theta_arr[m-1, :]
        theta_tilde = theta_arr[m-1, :]
        r_tilde = r0
        L_arr[m-1] = max(1, round(lambda / eps_arr[m-1]))
        for i in 1:L_arr[m-1]
            theta_tilde, r_tilde = leapfrog(likeli, theta_tilde, r_tilde, eps_arr[m-1])
        end
        
        accept = get_accept(likeli, theta_tilde, r_tilde, theta_arr[m-1], r0)
        accept_arr[m-1] = accept
        
        if rand() < accept  # metropolis correction
            theta_arr[m, :] = theta_tilde
            # r_m = -r_tilde
        end
        
        if m <= M_adapt  # warmup
            H_arr[m] = (1 - 1/(m+t0)) * H_arr[m-1] + 1/(m+t0) * (delta - accept)
            eps_arr[m] = exp(mu - sqrt(m) / gamma * H_arr[m])
            eps_bar_arr[m] = exp(m^(-kappa) * log(eps_arr[m]) + (1-m^(-kappa)) * log(eps_bar_arr[m-1]))
        else
            eps_arr[m] = eps_bar_arr[M_adapt]
        end
    end
    return Dict(:posterior => theta_arr, :eps_arr => eps_arr, :eps_bar_arr => eps_bar_arr,
                :accept_arr => accept_arr, :L_arr => L_arr)
end
