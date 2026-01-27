import torch
from torch import nn
import torch.nn.functional as F
import math

"""
    Two sample estimate of the MMD^2 distance
    Orignial tensorflow code: https://github.com/tolstikhin/wae/blob/master/wae.py#L233
"""
def mmd_penalty(z_hat: torch.Tensor, 
                prior_dist: torch.distributions, 
                kernel: str="IMQ", 
                sigma2_p: float=1.0, 
                discriminator: nn.Module=None,
                adversarial: bool=False, 
                lambda_gp: float=None,):
    """
    Computes the MMD^2 distance between a batch of samples z_hat and samples from prior_dist, MMD^2(Q_Z, P_Z),
    using either the RBF or IMQ kernel.
    """
    z = prior_dist.rsample(z_hat.size()).type_as(z_hat)
    n = z_hat.shape[0]
    half_size = int((n * n - n)/2)

    norms_z = z.pow(2).sum(dim=1, keepdims=True)
    dots_z = torch.mm(z, z.t())
    dists_z = (norms_z + norms_z.t() - 2. * dots_z).abs()

    norms_zh = z_hat.pow(2).sum(dim=1, keepdims=True)
    dots_zh = torch.mm(z_hat, z_hat.t())
    dists_zh = (norms_zh + norms_zh.t() - 2. * dots_zh).abs()

    dots = torch.mm(z_hat, z.t())
    dists = (norms_zh + norms_z.t() - 2. * dots).abs()

    if kernel == "RBF":
        # adaptive sigma2
        sigma2_k = torch.topk(dists_z.reshape(-1), half_size)[0][-1]
        sigma2_k = sigma2_k + torch.topk(dists_zh.reshape(-1), half_size)[0][-1]

        res1 = torch.exp(-dists_zh/2./sigma2_k)
        res1 = res1 + torch.exp(-dists_z/2./sigma2_k)
        res1 = torch.mul(res1, 1. - torch.eye(n).type_as(res1))
        res1 = res1.sum() / (n*n-n)
        res2 = torch.exp(-dists/2./sigma2_k)
        res2 = res2.sum()*2./(n*n)
        stat = res1 - res2
    elif kernel == "IMQ":
        Cbase = 2 * z.shape[1] * sigma2_p
        stat = 0.0
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + dists_z) + C / (C + dists_zh)
            res1 = torch.mul(res1, 1. - torch.eye(n).type_as(res1))
            res1 = res1.sum() / (n*n-n)
            res2 = C / (C + dists)
            res2 = res2.sum()*2./(n*n)
            stat = stat + res1 - res2
    return stat

def wgan_penalty(z_hat: torch.Tensor, 
                 prior_dist: torch.distributions,
                 discriminator: nn.Module,
                 adversarial: bool=False,
                 lambda_gp: float=5.0,
                 ):
    """
    Computes the 1-Wasserstein distance W_1(Q_Z, P_Z) between a batch of samples z_hat and samples from prior_dist using WGAN or WGAN-LP.
    1. If adversarial is False, computes the negative critic score on z_hat.
    2. If adversarial is True, computes the WGAN-GP loss using z_hat and samples from prior_dist.
    3. If lambda_gp > 0, includes the gradient penalty term.
    """
    qz = discriminator(z_hat)
    if adversarial:
        z_prior = prior_dist.rsample(z_hat.size()).type_as(z_hat)
        pz = discriminator(z_prior)
        grad_penalty = 0.0
        if lambda_gp > 0.0:
            grad_penalty = gradient_penalty(z_prior, z_hat, discriminator)
        return -torch.mean(pz) + torch.mean(qz) + lambda_gp * grad_penalty
    return -torch.mean(qz)

def gradient_penalty(x: torch.Tensor, y: torch.Tensor, f: torch.nn.Module):
    """
    Original code: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py#L119
    """
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1) # B, 1, 1, ..., 1
    alpha = torch.rand(shape).type_as(x)
    z = x + alpha * (y - x)
    # gradient penalty
    z = torch.autograd.Variable(z, requires_grad=True).type_as(x)
    o = f(z)
    g = torch.autograd.grad(
        outputs=o, 
        inputs=z, 
        grad_outputs=(torch.ones(o.size())).type_as(x), 
        create_graph=True,
        retain_graph=True
    )[0].view(z.size(0), -1)
    # g = grad(o, z, grad_outputs=(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
    gp = ((nn.functional.relu(g.norm(p=2, dim=1) - 1.0))**2).mean()
    return gp

def w1_penalty(z_hat: torch.Tensor, 
               prior_dist: torch.distributions,
               discriminator: nn.Module=None,
               adversarial: bool=False,
               lambda_gp: float=None,
                 ):
    """
    Computes an estimate of the 1-Wasserstein distance W_1(Q_Z, P_Z) by sorting a batch of sample z_hat.
    W_1(Q_Z, P_Z) = \int |F_{Q_Z}(t) - F_{P_Z}(t)| dt
    1. Sort z_hat to get empirical quantiles of Q_Z.
    2. Compute prior quantiles using the inverse CDF of prior_dist.
    3. Compute the smooth L1 loss between the two sets of quantiles as an estimate of W_1(Q_Z, P_Z).
    """
    z_sorted, _ = torch.sort(z_hat.flatten())
    n = z_hat.size(0)
    uniform_grid = (torch.arange(n, dtype=z_hat.dtype, device=z_hat.device) + .5) / n
    prior_quantiles = prior_dist.icdf(uniform_grid).detach()

    penalty = nn.functional.smooth_l1_loss(z_sorted, prior_quantiles, reduction='mean', beta=0.01)
    return penalty

"""
    f-divergence penalties
"""

def fgan_js_penalty(z_hat: torch.Tensor, 
                prior_dist: torch.distributions, 
                discriminator: nn.Module, 
                adversarial: bool=False,
                lambda_gp: float=None,
               ):
    """
    Jensen-Shannon divergence penalty using f-GAN formulation.
    D_JS(Q_Z || P_Z) = max_T {E_{Q_Z}[log sigma(T(z))] + E_{P_Z}[log (1 - sigma(T(z)))]}
    where sigma is the sigmoid function and T is the discriminator network.

    1. If adversarial is False, computes the supremand over Q_Z only.
    2. If adversarial is True, computes the full JS divergence using samples from both Q_Z and P_Z.
    """
    
    qz = discriminator(z_hat)
    # qz = torch.log(torch.exp(qz)+1) - torch.log(torch.tensor(2))
    qz = F.softplus(qz) - math.log(2)

    if adversarial:
        z_prior = prior_dist.rsample(z_hat.size()).type_as(z_hat)
        pz = discriminator(z_prior)
        # pz = torch.log(torch.tensor(2)) - torch.log(1+torch.exp(-pz))
        pz = math.log(2) - F.softplus(-pz)
        return torch.mean(qz) - torch.mean(pz)
    return -torch.mean(qz)

def fgan_kl_penalty(z_hat: torch.Tensor, 
                prior_dist: torch.distributions, 
                discriminator: nn.Module, 
                adversarial: bool=False,
                lambda_gp: float=None,
               ):
    """
    Kullback-Leibler divergence penalty using f-GAN formulation.
    D_KL(Q_Z || P_Z) = max_T E_{Q_Z}[T(z)] - E_{P_Z}[exp(T(z) - 1)]
                     = max_T E_{Q_Z}[T(z)+1] - E_{P_Z}[exp(T(z))]
    where T is the discriminator network.
    """
    
    qz = -discriminator(z_hat)-1
    
    if adversarial:
        z_prior = prior_dist.rsample(z_hat.size()).type_as(z_hat)
        pz = -torch.exp(discriminator(z_prior))
        return torch.mean(qz) - torch.mean(pz)
    return -torch.mean(qz)

def fgan_reverse_kl_penalty(z_hat: torch.Tensor, 
                prior_dist: torch.distributions, 
                discriminator: nn.Module, 
                adversarial: bool=False,
                lambda_gp: float=None,
               ):
    """
    Reverse Kullback-Leibler divergence penalty using f-GAN formulation.
    D_KL(P_Z || Q_Z) = max_T E_{P_Z}[T(z)] - E_{Q_Z}[exp(T(z)-1)]
    where T is the discriminator network.

    For stable training, we reparameterize T as T = 10 * tanh(D) - 1, where D is the output of the discriminator network.
    """
    qz = torch.exp(1e1*torch.tanh(discriminator(z_hat))-1)
    if adversarial:
        z_prior = prior_dist.rsample(z_hat.size()).type_as(z_hat)
        pz = 1e1*torch.tanh(discriminator(z_prior))
        return torch.mean(qz) - torch.mean(pz)
    return -torch.mean(qz)

def fgan_pearson_penalty(z_hat: torch.Tensor, 
                prior_dist: torch.distributions, 
                discriminator: nn.Module, 
                adversarial: bool=False,
                lambda_gp: float=None,
               ):
    """
    Pearson chi-squared divergence penalty using f-GAN formulation.
    D_Pearson(Q_Z || P_Z) = D_Neyman(P_Z || Q_Z) 
                          = max_T {E_{P_Z}[1 - exp(T(z))] - E_{Q_Z}[-2 exp(T(z)/2) + 2]}
    where T is the discriminator network.
    """
    qz = discriminator(z_hat)
    qz = -2 * torch.exp(qz/2) + 2

    if adversarial:
        z_prior = prior_dist.rsample(z_hat.size()).type_as(z_hat)
        pz = discriminator(z_prior)
        pz = -torch.exp(pz) + 1
        return torch.mean(qz) - torch.mean(pz)
    return -torch.mean(qz)

def fgan_neyman_penalty(z_hat: torch.Tensor, 
                prior_dist: torch.distributions, 
                discriminator: nn.Module, 
                adversarial: bool=False,
                lambda_gp: float=None,
               ):
    """
    Neyman chi-squared divergence penalty using f-GAN formulation.
    D_Neyman(Q_Z || P_Z) = D_Pearson(P_Z || Q_Z) 
                         = max_T {E_{P_Z}[T(z)] - E_{Q_Z}[(T(z))^2/4 + T(z)]}
    where T is the discriminator network.

    For stable training, we reparameterize T as T = 10*tanh(D), where D is the output of the discriminator network.
    """

    qz = 1e1*torch.tanh(discriminator(z_hat))
    qz = (qz**2) * .25 + qz
    
    if adversarial:
        z_prior = prior_dist.rsample(z_hat.size()).type_as(z_hat)
        pz = 1e1*torch.tanh(discriminator(z_prior))
        return torch.mean(qz) - torch.mean(pz)
    return -torch.mean(qz)

def fgan_sqHellinger_penalty(z_hat: torch.Tensor, 
                prior_dist: torch.distributions, 
                discriminator: nn.Module, 
                adversarial: bool=False,
                lambda_gp: float=None,
               ):
    """
    Squared Hellinger divergence penalty using f-GAN formulation.
    D_sqHellinger(Q_Z || P_Z) = D_sqHellinger(P_Z || Q_Z)
                              = max_T {E_{P_Z}[1 - exp(T(z))] - E_{Q_Z}[exp(-T(z))-1]}
    where T is the discriminator network.
    """
    
    qz = discriminator(z_hat)
    qz = torch.exp(-qz) - 1

    if adversarial:
        z_prior = prior_dist.rsample(z_hat.size()).type_as(z_hat)
        pz = discriminator(z_prior)
        pz = -torch.exp(pz) + 1
        return torch.mean(qz) - torch.mean(pz)
    return -torch.mean(qz)