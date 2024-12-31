import torch
from torch import nn, optim
import torch.nn.functional as F

from networks import *

class Autoencoder(nn.Module):
    def __init__(self, 
                 num_input_channels: int, 
                 base_channel_size: int, 
                 latent_dim: int,
                 learning_rate: float,
                 act_ftn: nn.Module = nn.LeakyReLU,
                 encoder: Encoder = Encoder,
                 decoder: Decoder = Decoder):
        super().__init__()

        self.encoder = encoder(num_input_channels, base_channel_size, latent_dim, act_ftn)
        self.decoder = decoder(num_input_channels, base_channel_size, latent_dim, act_ftn)
        self.lr = learning_rate

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def get_losses(self, x: torch.Tensor):
        x_hat = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x, reduction='none').sum(dim=[1,2,3]).mean(dim=[0])
        return {'recon_loss': recon_loss}
    
    def get_optimizers(self):
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr
        )

        scheduler = None
        return {"optimizer": optimizer, "scheduler": scheduler}