import torch
from torch import nn
from networks.base import Encoder, Decoder, Discriminator

# MNIST dataset
class Encoder_mnist(Encoder):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_ftn: nn.Module = nn.LeakyReLU):
        super().__init__(input_dim=1, latent_dim=latent_dim)

        n_channels = base_channel_size
        hidden_dim = 128
        self.model = nn.Sequential(
            nn.Conv2d(num_input_channels, n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels),
            act_ftn(),

            nn.Conv2d(n_channels, 2*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*n_channels),
            act_ftn(),

            nn.Conv2d(2*n_channels, 4*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*n_channels),
            act_ftn(),

            nn.Flatten(),
            nn.Linear(4*4*4*n_channels, hidden_dim),
            act_ftn(),
            # nn.Linear(hidden_dim, hidden_dim),
            # act_ftn(),
            nn.Linear(hidden_dim, latent_dim)
            # nn.Linear(4*4*4*n_channels, latent_dim)
        )

class Decoder_mnist(Decoder):
    def __init__(self, num_output_channels: int, base_channel_size: int, latent_dim: int, act_ftn: nn.Module = nn.LeakyReLU):
        super().__init__(output_dim=1, latent_dim=latent_dim)

        n_channels = base_channel_size
        hidden_dim = 128
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            act_ftn(),
            # nn.Linear(hidden_dim, hidden_dim),
            # act_ftn(),
            nn.Linear(hidden_dim, 4*4*4*n_channels),
            act_ftn(),
            # nn.Linear(latent_dim, 4*4*4*n_channels),
            # act_ftn(),

            nn.Unflatten(1, (4*n_channels, 4, 4)),
            nn.ConvTranspose2d(4*n_channels, 2*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*n_channels),
            act_ftn(),

            nn.ConvTranspose2d(2*n_channels, n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels),
            act_ftn(),

            nn.ConvTranspose2d(n_channels, num_output_channels, 4, 2, 1),
            # nn.Tanh(),
        )
        self.last_act = nn.Tanh()

class Discriminator_mnist(Discriminator):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__(latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

# CelebA dataset
class Encoder_celeba(Encoder):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_ftn: nn.Module = nn.LeakyReLU):
        super().__init__(input_dim=1, latent_dim=latent_dim)

        n_channels = base_channel_size
        hidden_dim = 64
        self.model = nn.Sequential(
            nn.Conv2d(num_input_channels, n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels),
            act_ftn(),

            nn.Conv2d(n_channels, 2*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*n_channels),
            act_ftn(),

            nn.Conv2d(2*n_channels, 4*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*n_channels),
            act_ftn(),

            nn.Conv2d(4*n_channels, 8*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8*n_channels),
            act_ftn(),

            nn.Flatten(),
            # nn.Linear(4*4*8*n_channels, hidden_dim),
            # act_ftn(),

            # nn.Linear(hidden_dim, hidden_dim),
            # act_ftn(),

            # nn.Linear(hidden_dim, latent_dim)
            nn.Linear(4*4*8*n_channels, latent_dim),


            ### Structure from the WAE paper
            # nn.Conv2d(num_input_channels, n_channels, 5, 2, 2),
            # nn.BatchNorm2d(n_channels),
            # act_ftn(),

            # nn.Conv2d(n_channels, 2*n_channels, 5, 2, 2),
            # nn.BatchNorm2d(2*n_channels),
            # act_ftn(),

            # nn.Conv2d(2*n_channels, 4*n_channels, 5, 2, 2),
            # nn.BatchNorm2d(4*n_channels),
            # act_ftn(),

            # nn.Conv2d(4*n_channels, 8*n_channels, 5, 2, 2),
            # nn.BatchNorm2d(8*n_channels),
            # act_ftn(),

            # nn.Flatten(),
            # nn.Linear(4*4*8*n_channels, latent_dim),
        )

class Decoder_celeba(Decoder):
    def __init__(self, num_output_channels: int, base_channel_size: int, latent_dim: int, act_ftn: nn.Module = nn.LeakyReLU):
        super().__init__(output_dim=1, latent_dim=latent_dim)

        n_channels = base_channel_size
        hidden_dim = 64
        self.model = nn.Sequential(
            # nn.Linear(latent_dim, hidden_dim),
            # act_ftn(),

            # nn.Linear(hidden_dim, hidden_dim),
            # act_ftn(),

            # nn.Linear(hidden_dim, 4*4*8*n_channels),
            # act_ftn(),
            
            nn.Linear(latent_dim, 4*4*8*n_channels),
            act_ftn(),

            nn.Unflatten(1, (8*n_channels, 4, 4)),
            nn.ConvTranspose2d(8*n_channels, 4*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*n_channels),
            act_ftn(),

            nn.ConvTranspose2d(4*n_channels, 2*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*n_channels),
            act_ftn(),

            nn.ConvTranspose2d(2*n_channels, n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels),
            act_ftn(),

            nn.ConvTranspose2d(n_channels, num_output_channels, 4, 2, 1),
            # nn.Tanh(),


            ### Structure from the WAE paper
            # nn.Linear(latent_dim, 8*8*8*n_channels),
            # act_ftn(),

            # nn.Unflatten(1, (8*n_channels, 8, 8)),
            # nn.ConvTranspose2d(8*n_channels, 4*n_channels, 5, 2, 2, 1),
            # nn.BatchNorm2d(4*n_channels),
            # act_ftn(),

            # nn.ConvTranspose2d(4*n_channels, 2*n_channels, 5, 2, 2, 1),
            # nn.BatchNorm2d(2*n_channels),
            # act_ftn(),

            # nn.ConvTranspose2d(2*n_channels, n_channels, 5, 2, 2, 1),
            # nn.BatchNorm2d(n_channels),
            # act_ftn(),

            # nn.ConvTranspose2d(n_channels, num_output_channels, 1, 1),
        )
        self.last_act = nn.Tanh()


'''
    Discriminator on the latent space
'''
class Discriminator_celeba(Discriminator):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__(latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

'''
    Discriminator on the image space
    Reference: iWGAN paper; https://arxiv.org/abs/2109.05652
'''
class Discriminator_image_celeba(Discriminator):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__(latent_dim)

        self.model = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 5, 2, 2),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(hidden_dim, 2*hidden_dim, 5, 2, 2),
            nn.InstanceNorm2d(2*hidden_dim, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(2*hidden_dim, 4*hidden_dim, 5, 2, 2),
            nn.InstanceNorm2d(4*hidden_dim, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(4*hidden_dim, 8*hidden_dim, 5, 2, 2),
            nn.InstanceNorm2d(8*hidden_dim, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(8*hidden_dim, 1, 4),
            nn.Flatten(),
        )