from typing import Callable
import torch
import numpy as np
from TruncNormal import TruncMultivariateNormal, PushforwardTruncMultivariateNormal
import matplotlib.pyplot as plt

def init_distributions(dim: int = None, optimal_encoder: Callable=None):
    
    assert dim is not None, "Dimension is not specified."
    # truncated interval
    boundary_pts = torch.randn((dim, 2)).abs().exp()
    boundary_pts[:, 0] *= -1.0
    a, b = boundary_pts[:, 0], boundary_pts[:, 1]

    # mean (near midpoint of the interval),
    # covariance (small correlation)
    L_rand = generate_positive_definite_cholesky(dim)
    L_rand[1, 0] *= 0.2
    loc_rand = (a + b) * 0.5 + torch.randn((dim, )) * 1e-2

    param_str =  f"P_X = TruncMultivariateNormal(mean = {loc_rand.numpy()}, cov = {(L_rand @ L_rand.t()).view(-1).numpy()}), support ="
    for i in range(dim):
        param_str += f"[{a[i].item():.5f}, {b[i].item():.5f}], "
    print(param_str)

    data_dist = TruncMultivariateNormal(loc_rand, L_rand, a, b)

    # Prior distribution: q*#P_X
    prior_dist = PushforwardTruncMultivariateNormal(loc_rand, L_rand, a, b, optimal_encoder)
        
    return data_dist, prior_dist

def generate_positive_definite_cholesky(dim: int, eps: float=1e-6):
    A = torch.randn(dim, dim)
    A = A @ A.t() # make symmetric

    # Add a small epsilon to the diagonal to ensure positive definiteness
    A += eps * torch.eye(dim)

    L = torch.linalg.cholesky(A)
    return L

def wasserstein_distance(mu: torch.distributions=None, potential=None):
    n_samples = 100000
    n_iter = 10
    
    wasserstein_dist_lst = []
    for _ in range(n_iter):
        sample = mu.rsample((n_samples, ))
        sample.requires_grad_()
        transport_sample = potential(sample)
        wasserstein_dist_lst.append(
            (transport_sample - sample).pow(2).sum(dim=1).mean().item()
        ) # 2-Wasserstein metric
    wasserstein_dist_ndarray = np.array(wasserstein_dist_lst)
    print(f"W(P_X, P_G) mean: {wasserstein_dist_ndarray.mean():.6f}, std: {wasserstein_dist_ndarray.std():.6f}")
    return wasserstein_dist_ndarray.mean()

def plot_losses(figpath: str, arr_recon: list, arr_penalty: list, coef: float, true_val: float, final_recon: float, fig_title=""):
    plt.figure(figsize = (15, 4))

    plt.subplot(131)
    plt.plot([arr_recon[i] + coef * arr_penalty[i] for i in range(len(arr_recon))])
    plt.axhline(true_val, color='r')
    plt.title("Objective function")
    plt.xlabel("epoch")

    plt.subplot(132)
    plt.plot(arr_recon)
    plt.axhline(true_val, color='r')
    plt.title(f"Reconstruction loss = {final_recon:.5f}, Ground truth = {true_val:.5f}")
    plt.xlabel("epoch")

    plt.subplot(133)
    plt.plot(arr_penalty)
    plt.title("Penalty")
    plt.xlabel("epoch")

    plt.suptitle(fig_title)
    plt.savefig(f"{figpath}_losses.png")
    plt.close()