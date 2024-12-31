import numpy as np

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
        import scipy
        q_vec = np.linspace(0.0001, 0.9999, 499995)
        quantile1 = mu1.icdf(torch.from_numpy(q_vec).type_as(mu1.a)).cpu().numpy()
        quantile2 = mu2.icdf(torch.from_numpy(q_vec).type_as(mu2.a)).cpu().numpy()
        
        return scipy.integrate.simpson(y=np.power(quantile2 - quantile1, 2), x=q_vec)
    else:
        raise NotImplementedError(f"{dist_family} is not implemented")
    
"""
    the p-value of Kolmogorov-Smirnov test between the aggregated posterior Q_Z and the prior distribution P_Z
"""
def ks_test(encoder: nn.Module, data_dist: torch.distributions, prior_dist: torch.distributions, iter_test: int=1):
    device = next(encoder.parameters()).device
    n_test = 25000
    from scipy.stats import ks_1samp
    prior_cdf = lambda z: prior_dist.cdf(torch.from_numpy(z)).numpy()

    if iter_test == 1:
        # samples from Q_Z
        x_test = data_dist.rsample((n_test, 1)).to(device)
        with torch.no_grad():
            z_encoded = encoder(x_test).squeeze().cpu() 
        ks_result = ks_1samp(z_encoded.numpy(), prior_cdf)
        return ks_result.statistic, ks_result.pvalue
    else:
        tot_ksstat, tot_pvalue = 0.0, 0.0
        pass_cnt = 0
        for _ in range(iter_test):
            # samples from Q_Z
            x_test = data_dist.rsample((n_test, 1)).to(device)
            with torch.no_grad():
                z_encoded = encoder(x_test).squeeze().cpu() 
            ks_result = ks_1samp(z_encoded.numpy(), prior_cdf)
            if ks_result.pvalue > 0.05:
                pass_cnt += 1
            tot_ksstat += ks_result.statistic
            tot_pvalue += ks_result.pvalue
        return tot_ksstat / iter_test, tot_pvalue / iter_test, pass_cnt / iter_test

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
def plot_encoder(figpath: str, encoder: nn.Module, data_dist: torch.distributions, prior_dist: torch.distributions, arr_ks_stat: list, arr_ks_pval: list):
    device = next(encoder.parameters()).device
    n_test = 5000
    
    # Samples from Q_Z
    x_test = data_dist.rsample((n_test, 1)).to(device)
    with torch.no_grad():
        z_encoded = encoder(x_test).squeeze().cpu()

    plt.figure(figsize = (15, 4))
    plt.subplot(131)
    plt.plot(z_encoded, prior_dist.cdf(z_encoded), color="r", label="data")
    plt.plot(sorted(z_encoded), np.arange(n_test)/n_test, color="b", label="encoder")
    plt.legend() 
    plt.title("Encoder")

    plt.subplot(132)
    plt.plot(arr_ks_stat)
    plt.title("KS-test statistic")
    plt.xlabel("epoch")

    plt.subplot(133)
    plt.plot(arr_ks_pval)
    plt.title("KS-test p-value")
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
def save_data(figpath: str, 
            epoch: int, 
            arr_obj: list, 
            arr_recon: list, 
            arr_penalty: list, 
            arr_ks_stat: list, 
            arr_ks_pval: list, 
            arr_pass_rate: list, 
            arr_neg_pen: list):
    
    with open(figpath+"_data.txt", "w") as file:
        file.write("epoch | obj        | recon      | penalty    | ks_stat    | ks_pval    | pass_pval  | neg_aux_count \n")
        for i in range(epoch):
            file.write(
                f"{(i+1):>5d} | {arr_obj[i]:<10.5g} | {arr_recon[i]:<10.5g} | {arr_penalty[i]:<10.5g} " \
                + f"| {arr_ks_stat[i]:<10.5g} | {arr_ks_pval[i]:<10.5g} | {arr_pass_rate[i]:<10.5g} | {arr_neg_pen[i]:3d}\n"
            )
        