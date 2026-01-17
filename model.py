import torch
import torch.nn as nn 
from utils import get_encoder_activation

class ResBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1): 
        super(ResBlock, self).__init__() 
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.activation = nn.LeakyReLU(0.2, inplace=True) 

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_blocks=1):
        super().__init__()
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResNetEncoder(nn.Module):
    def __init__(self, hidden_dim, channels, image_size, latent_dim, prior_type):
        super().__init__()
        h = hidden_dim
        self.feature_extractor = nn.Sequential( 
            nn.Conv2d(channels, h, 7, 2, 3, bias=False),
            nn.BatchNorm2d(h), 
            nn.LeakyReLU(0.2, inplace=True), 
            ResLayer(h, h * 2, stride=2, num_blocks=2), 
            ResLayer(h * 2, h * 4, stride=2, num_blocks=2), 
            ResLayer(h * 4, h * 8, stride=2, num_blocks=2), 
            ResLayer(h * 8, h * 16, stride=2, num_blocks=2), 
        )

        self.final_spatial_dim = image_size // 32
        self.flatten_dim = (h * 16) * (self.final_spatial_dim ** 2)
        
        self.fc = nn.Linear(self.flatten_dim, latent_dim)
        self.activation = get_encoder_activation(prior_type, latent_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.activation(x)

class ResNetDecoder(nn.Module): 
    def __init__(self, hidden_dim, image_size, latent_dim, channels): 
        super().__init__() 
        h = hidden_dim
        self.start_spatial_dim = image_size // 32
        self.linear_dim = (h * 16) * (self.start_spatial_dim ** 2)

        self.fc = nn.Linear(latent_dim, self.linear_dim)
        self.unflatten = nn.Unflatten(1, (h * 16, self.start_spatial_dim, self.start_spatial_dim))
        
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ResLayer(h * 16, h * 8, num_blocks=2),
            
            nn.Upsample(scale_factor=2),
            ResLayer(h * 8, h * 4, num_blocks=2),
            
            nn.Upsample(scale_factor=2),
            ResLayer(h * 4, h * 2, num_blocks=2),
            
            nn.Upsample(scale_factor=2),
            ResLayer(h * 2, h, num_blocks=2),

            nn.Upsample(scale_factor=2),
            ResBlock(h, h),
            nn.Conv2d(h, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        return self.main(x)

class Discriminator(nn.Module): 
    def __init__(self, latent_dim): 
        super().__init__() 
        h = latent_dim
        self.main = nn.Sequential( 
            nn.utils.spectral_norm(nn.Linear(h, h * 8)), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(h * 8, h * 8)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(h * 8, h * 8)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(h * 8, h * 8)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(h * 8, 1))
        )

    def forward(self, x):
        return self.main(x)