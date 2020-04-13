"""
theta_arr: Array, (chains, draws)
"""
function R_hat(theta_arr)
    M, N = size(theta_arr)
    theta_mean = mean(theta_arr, dims=2)
    theta_mean_mean = mean(theta_mean)
    B = N/(M-1) * sum((theta_mean .- theta_mean_mean).^2)
    s2 = 1/(N-1) * sum((theta_arr .- theta_mean).^2, dims=2)
    W = mean(s2)
    var_plus = (N-1)/N * W + 1/N * B
    R_hat = sqrt(var_plus/W)
    return R_hat
end

function split_chains(theta_arr)
    M, N = size(theta_arr)
    split_N = N ÷ 2
    split_theta_arr = similar(theta_arr, M*2, split_N)
    for i in 1:M
        split_theta_arr[2*i-1, :] = theta_arr[i, 1:split_N]
        split_theta_arr[2*i, :] = theta_arr[i, N-split_N+1:N]
    end
    return split_theta_arr
end

"""
theta_arr: Array, (chains, draws)
"""
function split_R_hat(theta_arr)
    return R_hat(split_chains(theta_arr))
end

"""
theta_arr: Array, (chains, draws)
"""
function ess(theta_arr)
    M, N = size(theta_arr)
    theta_mean = mean(theta_arr, dims=2)
    theta_mean_mean = mean(theta_mean)
    B = N/(M-1) * sum((theta_mean .- theta_mean_mean).^2)
    s2 = 1/(N-1) * sum((theta_arr .- theta_mean).^2, dims=2)
    W = mean(s2)
    var_plus = (N-1)/N * W + 1/N * B
    
    rho_chains = autocor(theta_arr', 0:size(theta_arr, 2)-1)' # (chains, draws)
    global rho = 1 .- (W .- mean(s2 .* rho_chains, dims=1)) ./ var_plus # (1, draws)
    
    tau = 1.
    t = 1
    P_old = Inf
    while 2*t+1 < size(rho, 2)
        P = rho[1, 2*t] + rho[1, 2*t + 1]
        
        # Initial positive sequence estimators
        if P < 0
            break
        end
        
        # Initial monotone sequence
        if P > P_old
            P = P_old
        else
            P_old = P
        end
        
        tau += P*2
        t += 1
    end
    
    return M * N / tau
end

"""
posterior: [chains, draws, parameters]

Stan equivalent reference output:
           mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
mu         7.96    0.18   5.38   -2.5   4.53   7.86  11.24  19.07    866    1.0
"""
function mcmc_summary(posterior)
    data_mat = Matrix(undef, size(posterior, 3), 10)
    for pid in 1:size(posterior, 3)
        post = posterior[:, :, pid]
        n_eff = ess(post)
        Rhat = split_R_hat(post)
        _mean = mean(post)
        se_mean = mean(post) / sqrt(n_eff)
        _sd = std(post)
        q_arr = quantile(vec(post), [0.025, 0.25, 0.5, 0.75, 0.9775])
        row = [_mean, se_mean, _sd, q_arr..., n_eff, Rhat]
        data_mat[pid, :] = row
    end
    df = DataFrame(data_mat, [:mean, :se_mean, :sd, Symbol("q2.5"), :q25, :q50, :q75, Symbol("q97.75"), :n_eff, :Rhat])
end
