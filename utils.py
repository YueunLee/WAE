import torch

class BasePrior:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def sample(self, batch_size, device):
        raise NotImplementedError

class NormalPrior(BasePrior):
    """Standard Normal Distribution: N(0, I)"""
    def sample(self, batch_size, device):
        return torch.randn(batch_size, self.latent_dim, device=device)

class UniformPrior(BasePrior):
    """Uniform Distribution: U(0, 1)"""
    def sample(self, batch_size, device):
        return torch.rand(batch_size, self.latent_dim, device=device)

class LogitNormalPrior(BasePrior):
    """Logit-Normal Distribution derived from N(0, I)."""
    def sample(self, batch_size, device):
        return torch.sigmoid(torch.randn(batch_size, self.latent_dim, device=device))

def get_prior(prior_type, latent_dim):
    """Prior Factory Function"""
    p_type = prior_type.lower()
    if p_type == 'normal':
        return NormalPrior(latent_dim)
    elif p_type == 'uniform':
        return UniformPrior(latent_dim)
    elif p_type == 'logit_normal':
        return LogitNormalPrior(latent_dim)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

def get_encoder_activation(prior_type, latent_dim):
    p_type = prior_type.lower()
    if p_type == 'normal':
        return torch.nn.BatchNorm1d(latent_dim, affine=False)
    elif p_type == 'uniform':
        return torch.nn.Sigmoid()
    elif p_type == 'logit_normal':
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(latent_dim, affine=False),
            torch.nn.Sigmoid(),
        )
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

def get_transform(divergence_type):
    """return g_f (target transform) & -f^*(g_f) (basis transform)"""
    d_type = divergence_type.lower()
    if d_type == 'kl':
        target_transform = lambda x: x
        prior_transform = lambda x: -torch.exp(x-1)
    elif d_type == 'js':
        target_transform = lambda x: -torch.nn.functional.softplus(-x)
        prior_transform = lambda x: torch.nn.functional.softplus(-x)
    elif d_type == 'reverse_kl':
        target_transform = lambda x: -torch.exp(-x)
        prior_transform = lambda x: -x+1
    elif d_type == 'pearson':
        target_transform = lambda x: x
        prior_transform = lambda x: -0.25 * x.pow(2) - x
    elif d_type == 'neyman':
        target_transform = lambda x: 1-torch.exp(x)
        prior_transform = lambda x: -2+2*torch.exp(x/2)
    elif d_type == 'hellinger':
        target_transform = lambda x: 1-torch.exp(x)
        prior_transform = lambda x: 1-torch.exp(-x)
    else:
        raise ValueError(f"Unknown divergence type: {divergence_type}")

    return target_transform, prior_transform