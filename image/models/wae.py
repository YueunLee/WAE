from typing import List
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from networks import Encoder, Decoder, Discriminator
from utils import mmd_penalty
from models.base import Autoencoder
    
class WAE_MMD(Autoencoder):
    def __init__(self, 
                 num_input_channels: int, 
                 base_channel_size: int, 
                 latent_dim: int,
                 learning_rate: float,
                 sampler: str,
                 act_ftn: nn.Module = nn.LeakyReLU,
                 encoder: Encoder = Encoder,
                 decoder: Decoder = Decoder):
        super().__init__(num_input_channels, base_channel_size, latent_dim, learning_rate,
                         act_ftn, encoder, decoder)
        self.latent_dim = latent_dim
        self.lower_sampler = sampler.lower()
        if self.lower_sampler == "uniform":
            self.encoder.last_act = nn.Tanh()

    def first_operation(self, x: torch.Tensor):
        z = self.encoder(x)
        return z, x

    def get_losses(self, ingredient, aux: float = None):
        z, x = ingredient
        x_hat = self.decoder(z) # range: [-1.0, 1.0]
        recon_loss = F.mse_loss(x_hat, x, reduction='none').sum(dim=[1,2,3]).mean(dim=[0])
        
        if self.lower_sampler == "normal": # N(0, 1)
            z_prior = torch.randn_like(z)
        else: # U[-1,1]
            z_prior = torch.rand_like(z)
            z_prior = (z_prior - .5) / .5
        mmd = mmd_penalty(z, z_prior)
        
        return {'recon_loss': recon_loss, 'mmd_penalty': mmd}

    def get_optimizers(self):
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr
        )

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 0.5**(epoch>=30) * 0.2**(epoch>=50)
            )
        return {"optimizer": [optimizer], "scheduler": [scheduler]}


class WAE_GAN(Autoencoder):
    def __init__(self, 
                 num_input_channels: int, 
                 base_channel_size: int, 
                 latent_dim: int,
                 learning_rate: float,
                 adv_learning_rate: float,
                 sampler: str,
                 act_ftn: nn.Module = nn.LeakyReLU,
                 encoder: Encoder = Encoder,
                 decoder: Decoder = Decoder,
                 discriminator: Discriminator = Discriminator,                 
                 ):
        super().__init__(num_input_channels, base_channel_size, latent_dim, learning_rate,
                         act_ftn, encoder, decoder)
        self.latent_dim = latent_dim
        self.adv_lr = adv_learning_rate
        self.lower_sampler = sampler.lower()
        if self.lower_sampler == "uniform":
            self.encoder.last_act = nn.Tanh()
        # if exact:
        #     self.n_steps = 10 # the number of steps for the Newton-Raphson method
        self.discriminator = discriminator(latent_dim, hidden_dim=512)

    def first_operation(self, x: torch.Tensor):
        return self.encoder(x), x

    def get_adv_losses(self, ingredient):
        z, _ = ingredient
        if self.lower_sampler == "normal": # N(0,1)
            z_prior = torch.randn_like(z)
        else: # U(-1, 1)
            z_prior = torch.rand_like(z)
            z_prior = (z_prior - .5) / .5

        pz = self.discriminator(z_prior)
        qz = self.discriminator(z.detach())
        adv_loss = F.binary_cross_entropy_with_logits(pz, torch.ones_like(pz)) + \
            F.binary_cross_entropy_with_logits(qz, torch.zeros_like(qz))
        return {'adv_loss': adv_loss}

    def get_losses(self, ingredient, aux: float = None):
        z, x = ingredient
        x_hat = self.decoder(z)
        recon_loss = F.mse_loss(x_hat, x, reduction='none').sum(dim=[1,2,3]).mean(dim=[0])
        
        qz = self.discriminator(z)
        gan_penalty = F.binary_cross_entropy_with_logits(qz, torch.ones_like(qz))
        if aux is not None:
            gan_penalty *= (-aux).exp().item()

        return {'recon_loss': recon_loss, 'gan_penalty': gan_penalty}
    
    def get_optimizers(self):
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr
        )
        adv_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.adv_lr)

        # Manual scheduling
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 0.5**(epoch>=30) * 0.2**(epoch>=50)
            )
        adv_scheduler = optim.lr_scheduler.LambdaLR(
            adv_optimizer,
            lr_lambda=lambda epoch: 0.5**(epoch>=30) * 0.2**(epoch>=50)
            )
        
        return {"optimizer": [optimizer, adv_optimizer], "scheduler": [scheduler, adv_scheduler]}