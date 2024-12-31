import torch
from torch import nn

# base classes
class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.model = nn.Linear(input_dim, latent_dim)
        self.last_act = nn.Identity()

    def forward(self, x) -> torch.Tensor:
        return self.last_act(self.model(x))

class Decoder(nn.Module):
    def __init__(self, output_dim: int, latent_dim: int):
        super().__init__()
        self.model = nn.Linear(latent_dim, output_dim)
        self.last_act = nn.Identity()
    
    def forward(self, z):
        return self.last_act(self.model(z))

class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.model(x)