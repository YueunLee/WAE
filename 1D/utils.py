import numpy as np
import scipy

import torch
from torch import nn
from TruncNormal import TruncatedNormal

import matplotlib.pyplot as plt

"""
    Initialize the data distribution P_X and the model distribution P_G
"""
def init_distributions(dist_family: str, scaleX: float, scaleG: float, coefX: float, coefG: float):
    if dist_family == "truncated_normal":

        loc = torch.randn(1)
        scale = torch.randn(1).abs().exp()
        scale /= scaleX
        a, b = loc - coefX * scale, loc + coefX * scale
        
        print(
            f"P_X = TruncNormal(loc = {loc.item(): .5f}, scale = {scale.item(): .5f}, " \
            + f"support = [{a.item():.5f}, {b.item(): .5f}])"
        )        
        data_dist = TruncatedNormal(
            loc=loc.item(),
            scale=scale.item(), 
            a=a.item(), 
            b=b.item()
        )

        loc = torch.randn(1)
        scale = torch.randn(1).abs().exp()
        scale /= scaleG
        a, b = loc - coefG * scale, loc + coefG * scale
        print(
            f"P_G = TruncNormal(loc = {loc.item(): .5f}, scale = {scale.item(): .5f}, " \
            + f"support = [{a.item():.5f}, {b.item(): .5f}])"
        )
        gen_dist = TruncatedNormal(
            loc=loc.item(), 
            scale=scale.item(), 
            a=a.item(), 
            b=b.item()
        )
        
    else:
        raise NotImplementedError(f"{dist_family} is not implemented")
        
    return data_dist, gen_dist, scale.item(), coefG


"""
    define a decoder (pushforwarding U[0,1] -> P_G); assuming the prior distribution P_Z = U[0,1]
"""   
def init_decoder(z: torch.Tensor, dist: torch.distributions):
    if type(dist) is TruncatedNormal:
        return dist.icdf(z).reshape(z.size())
    else:
        raise NotImplementedError(f"{type(dist)} is not implemented")

"""
    Compute the 2-Wasserstein distance between two distributions;
        = Integral of squared distance of two quantile functions
"""
# square of 2-Wasserstein distance between two uniform distributions
def wasserstein_distance(mu1: torch.distributions, mu2: torch.distributions, dist_family: str):
    if dist_family == "truncated_normal":
        """
        approximation by quantile function.
        """
        q_vec = np.linspace(0.0001, 0.9999, 499995)
        quantile1 = mu1.icdf(torch.from_numpy(q_vec).type_as(mu1.a)).cpu().numpy()
        quantile2 = mu2.icdf(torch.from_numpy(q_vec).type_as(mu2.a)).cpu().numpy()
        
        return scipy.integrate.simpson(y=np.power(quantile2 - quantile1, 2), x=q_vec)
    else:
        raise NotImplementedError(f"{dist_family} is not implemented")

def sampled_wasserstein_distance(samples: np.ndarray):
    """ 
        Compute the empirical 1-Wasserstein distance between samples from an encoded distribution
        and the uniform distribution on [0,1].
        
        Uses the formula: W_1 ≈ (1/n) Σ |z_(i) - (2i-1)/(2n)|
        where z_(i) are sorted samples.
    """
    z_sorted = np.sort(samples)
    n = len(z_sorted)
    
    # Compute quantile points of uniform[0,1]: (2i-1)/(2n) for i=1,...,n
    uniform_quantiles = (2 * np.arange(1, n+1) - 1) / (2 * n)
    
    # Compute 1-Wasserstein distance
    w1 = np.mean(np.abs(z_sorted - uniform_quantiles))
    
    return w1

def equivalence_test(encoder: nn.Module, data_dist: torch.distributions, prior_dist: torch.distributions, alpha: float=0.05, n_samples: int=5000, iter_test: int = 1, margin: float=1e-2):
    device = next(encoder.parameters()).device
    prior_cdf = lambda z: prior_dist.cdf(torch.from_numpy(z)).numpy()

    if iter_test == 1:
        # samples from Q_Z
        x_test = data_dist.rsample((n_samples, 1)).to(device)
        with torch.no_grad():
            z_encoded = encoder(x_test).squeeze().cpu() 
        
        d_obs, _ = scipy.stats.ks_1samp(z_encoded.numpy(), prior_cdf)
        
        # Time-Uniform Radius (DKW inequality)
        radius = np.sqrt(np.log(2 / alpha) / (2 * n_samples))
        upper_bound = d_obs + radius
        is_equivalent = int(upper_bound < margin)
        return d_obs, upper_bound, is_equivalent
    else:
        tot_d_obs, tot_upper_bound, tot_is_equivalent = 0.0, 0.0, 0.0
        for _ in range(iter_test):
            # samples from Q_Z
            x_test = data_dist.rsample((n_samples, 1)).to(device)
            with torch.no_grad():
                z_encoded = encoder(x_test).squeeze().cpu() 
            
            d_obs, _ = scipy.stats.ks_1samp(z_encoded.numpy(), prior_cdf)

            # Time-Uniform Radius (DKW inequality)
            radius = np.sqrt(np.log(2 / alpha) / (2 * n_samples))
            upper_bound = d_obs + radius
            is_equivalent = int(upper_bound < margin)

            tot_d_obs += d_obs
            tot_upper_bound += upper_bound
            tot_is_equivalent += is_equivalent
        return tot_d_obs / iter_test, tot_upper_bound / iter_test, tot_is_equivalent / iter_test

"""
    Plot losses per epoch
"""
def plot_losses(figpath: str, arr_recon: list, arr_penalty: list, arr_obj: list, true_val: float, final_recon: float):
    plt.figure(figsize = (15, 4))

    plt.subplot(131)
    plt.plot(arr_obj)
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

    plt.savefig(f"{figpath}_losses.png")
    plt.close()
    
"""
    Plot the encoder function pushforwarding P_X vs. the prior P_Z
"""
def plot_encoder(
        figpath: str, 
        encoder: nn.Module, 
        data_dist: torch.distributions, 
        prior_dist: torch.distributions, 
        arr_w1: list,
        arr_equiv_stat: list, 
        arr_equiv_ubd: list,
    ):
    device = next(encoder.parameters()).device
    n_test = 5000
    
    # Samples from Q_Z
    x_test = data_dist.rsample((n_test, 1)).to(device)
    with torch.no_grad():
        z_encoded = encoder(x_test).squeeze().cpu()

    plt.figure(figsize = (20, 4))
    plt.subplot(141)
    plt.plot(z_encoded, prior_dist.cdf(z_encoded), color="r", label="data")
    plt.plot(sorted(z_encoded), np.arange(n_test)/n_test, color="b", label="encoder")
    plt.legend() 
    plt.title("Encoder")

    plt.subplot(142)
    plt.plot(arr_w1)
    plt.title("Latent W1 distance")
    plt.xlabel("epoch")

    plt.subplot(143)
    arr_equiv_stat = np.array(arr_equiv_stat)
    epochs = np.arange(1, len(arr_equiv_stat)+1)
    plt.plot(epochs[arr_equiv_stat != 0], arr_equiv_stat[arr_equiv_stat != 0], marker='o', markersize=3)
    plt.title("Equivalence test statistic")
    plt.xlabel("epoch")

    plt.subplot(144)
    arr_equiv_ubd = np.array(arr_equiv_ubd)
    epochs = np.arange(1, len(arr_equiv_ubd)+1)
    plt.plot(epochs[arr_equiv_ubd != 0], arr_equiv_ubd[arr_equiv_ubd != 0], marker='o', markersize=3)
    plt.axhline(0.02, color='r')
    plt.title("Equivalence test upper bound")
    plt.xlabel("epoch")
    
    plt.savefig(f"{figpath}_encoder.png")
    plt.close()

"""
    Plot the auxiliary variable
"""
def plot_auxiliary(figpath: str, arr_aux: list):
    plt.plot(arr_aux)
    plt.title("Auxiliary variable t")
    plt.savefig(figpath+"_auxiliary.png")
    plt.close()

"""
    Save the data as a file
"""
def save_data(
        figpath: str, 
        epoch: int, 
        arr_obj: list, 
        arr_recon: list, 
        arr_penalty: list, 
        arr_w1: list,
        arr_neg_pen: list,
        arr_equiv_stat: list, 
        arr_equiv_ubd: list, 
        arr_equivalence: list, 
    ):
    
    with open(figpath+"_data.txt", "w") as file:
        file.write("epoch | obj        | recon      | penalty    | latent_w1  | neg_aux_count | equiv_stat    | equiv_ubd    | equivalence  |  \n")
        for i in range(epoch):
            file.write(
                f"{(i+1):>5d} | {arr_obj[i]:<10.5g} | {arr_recon[i]:<10.5g} | {arr_penalty[i]:<10.5g} " \
                + f"| {arr_w1[i]:<10.5g} | {arr_neg_pen[i]:<3d} " \
                + f"| {arr_equiv_stat[i]:<10.5g} | {arr_equiv_ubd[i]:<10.5g} | {arr_equivalence[i]:<10.5g}\n"
            )
        