import torch
import numpy as np
from torch.distributions import Normal

def compute_kernel_matrix(x, y, sigma2):
    """
    Kernel matrix computation using Gaussian RBF kernel.
    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    # pairwise distance computation
    dist_sq = torch.cdist(x, y, p=2)**2
    return torch.exp(-dist_sq / (2 * sigma2))

def compute_multiscale_kernel(x, y, base_sigma, scales=[0.5, 1.0, 2.0, 4.0, 8.0]):
    """
    Multi-scale Gaussian Kernel
    K(x, y) = Mean_j [ exp(-||x-y||^2 / (2 * (sigma * scale_j)^2)) ]
    """
    dist_sq = torch.cdist(x, y, p=2)**2
    kernel_sum = 0.0
    
    for s in scales:
        sigma = base_sigma * s
        gamma = 1.0 / (2 * sigma**2)
        kernel_sum += torch.exp(-gamma * dist_sq)
        
    # 커널 값의 스케일을 유지하기 위해 평균 사용 (Sum 대신 Mean)
    return kernel_sum / len(scales)


def unbiased_mmd_squared_block(x_block, y_block, sigma2, multiscale=False):
    """
    Compute the unbiased estimator of MMD^2 for a single block.
    (Gretton et al., 2012, Lemma 6)
    """
    m = x_block.shape[0]
    
    if multiscale:
        compute_kernel_ftn = compute_multiscale_kernel
    else:
        compute_kernel_ftn = compute_kernel_matrix

    # Kernel Matrices
    K_xx = compute_kernel_ftn(x_block, x_block, sigma2)
    K_yy = compute_kernel_ftn(y_block, y_block, sigma2)
    K_xy = compute_kernel_ftn(x_block, y_block, sigma2)
    
    # 대각 성분(자기 자신과의 거리) 제거 (Unbiased estimator를 위해)
    # torch.diagonal은 view를 반환하므로 fill_로 수정 가능
    K_xx.fill_diagonal_(0)
    K_yy.fill_diagonal_(0)
    
    # Unbiased Statistic Formula
    # Term 1: E[k(x, x')] -> sum / (m * (m-1))
    # Term 2: E[k(y, y')] -> sum / (m * (m-1))
    # Term 3: E[k(x, y)]  -> sum / (m * m)  <-- XY는 대각 제거 안 함
    
    term_1 = K_xx.sum() / (m * (m - 1))
    term_2 = K_yy.sum() / (m * (m - 1))
    term_3 = K_xy.sum() / (m * m)
    
    return term_1 + term_2 - 2 * term_3


def mmd_batched_equivalence_test(
    z_hat: torch.Tensor, 
    z: torch.Tensor, 
    block_size: int, 
    delta: float, 
    alpha: float = 0.05,
    sigma2: float = None,
    multiscale: bool = False
):
    """
    B-test based MMD equivalence test.
    
    H0: MMD^2 >= delta
    HA: MMD^2 < delta
    
    Args:
        z_hat (Tensor): encoded latent vectors (shape: [N, D])
        z (Tensor): sampled vectors from prior (shape: [N, D])
        block_size (int): block size for the B-test, recommended sqrt(N) in the paper.
        delta (float): threshold for acceptable MMD^2 distance.
        alpha (float): significance level (default 0.05).
        sigma (float, optional): bandwidth for RBF kernel. If None, uses Median Heuristic.
        multiscale (bool): whether to use multiscale kernel.
        
    Returns:
        dict: test results including:
            - is_equivalent (bool): True if H0 is rejected (i.e., distributions are equivalent)
            - mmd_mean (float): estimated mean MMD^2
            - upper_bound (float): upper bound of the confidence interval
    """
    assert len(z_hat) == len(z), "Sample size must be the same for both distributions"
    
    n_samples = len(z_hat)
    n_blocks = n_samples // block_size
    
    if n_blocks < 2:
        raise ValueError(f"Number of samples({n_samples}) must be at least twice the block size({block_size}) to estimate variance.")
    
    # 1. Trim data to fit into full blocks
    z_hat = z_hat[:n_blocks * block_size]
    z = z[:n_blocks * block_size]
    
    # 2. Set sigma (bandwidth) - Median Heuristic
    if sigma2 is None:
        subset_size = min(len(z), 1000)
        # idx = np.random.choice(len(z), subset_size, replace=False)
        # z_sample = torch.cat([z_hat[idx], z[idx]], dim=0)
        z_sample = torch.cat([z_hat[:subset_size], z[:subset_size]], dim=0)
        
        dists = torch.cdist(z_sample, z_sample, p=2).pow(2)
        dists = dists[dists > 0]
        sigma2 = dists.median() if len(dists) > 0 else 1.0
    
    # 3. Compute MMD^2 in blocks (core of B-test)
    # Reshape data to [Block_Count, Block_Size, Dim]
    z_hat_blocks = z_hat.view(n_blocks, block_size, -1)
    z_blocks = z.view(n_blocks, block_size, -1)
    
    block_mmds = []
    for i in range(n_blocks):
        val = unbiased_mmd_squared_block(z_hat_blocks[i], z_blocks[i], sigma2, multiscale=multiscale)
        block_mmds.append(val)
        
    block_mmds = torch.stack(block_mmds)
    
    # 4. Statistical Estimates
    mean_mmd = torch.mean(block_mmds)
    
    # Sample Variance & Standard Error
    sample_variance = torch.var(block_mmds, unbiased=True)
    standard_error = torch.sqrt(sample_variance / n_blocks)
    
    # 5. Equivalence Test
    # To reject H0, the upper bound of the 95% CI must be less than delta.
    # z_score for one-sided test (e.g., 1.645 for alpha=0.05)
    normal_dist = Normal(0, 1)
    z_score = normal_dist.icdf(torch.tensor(1 - alpha))
    
    upper_bound = mean_mmd + z_score * standard_error
    
    reject_null = int((upper_bound < delta).item())
    
    return {
        "is_equivalent": reject_null,       # 1 if equivalent, 0 otherwise
        "mmd_mean": mean_mmd.item(),        # estimated MMD^2 mean
        "upper_bound": upper_bound.item(),  # Upper bound of the confidence interval
        # "std_error": standard_error.item(), # standard deviation of the estimate
        # "sigma2": sigma2,                   # used bandwidth
        # "n_blocks": n_blocks
    }

