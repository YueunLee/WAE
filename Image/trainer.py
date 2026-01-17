import torch 
import torch.nn as nn 
import torch.optim as optim 
import pytorch_lightning as pl 
import torchvision 
from torchmetrics.image.fid import FrechetInceptionDistance 
from model import ResNetEncoder, ResNetDecoder, Discriminator 
from prdc import compute_prdc
from utils import get_prior, get_transform

class WAE_GAN(pl.LightningModule): 
    def __init__(self, config): 
        super().__init__() 
        self.save_hyperparameters(config)
        self.cfg = config 
        
        self.automatic_optimization = False

        self.encoder = ResNetEncoder(
            hidden_dim=self.cfg['HIDDEN_DIM'],
            channels=self.cfg['CHANNELS'],
            image_size=self.cfg['IMAGE_SIZE'],
            latent_dim=self.cfg['LATENT_DIM'],
            prior_type=self.cfg['PRIOR_TYPE']
        )
        self.decoder = ResNetDecoder(
            hidden_dim=self.cfg['HIDDEN_DIM'],
            image_size=self.cfg['IMAGE_SIZE'],
            latent_dim=self.cfg['LATENT_DIM'],
            channels=self.cfg['CHANNELS']
        )
        self.discriminator = Discriminator(
            latent_dim=self.cfg['LATENT_DIM']
        )

        self.prior = get_prior(self.cfg['PRIOR_TYPE'], self.cfg['LATENT_DIM'])
        self.target_transform, self.prior_transform = get_transform(self.cfg['DIVERGENCE_TYPE'])
        
        self.fid = FrechetInceptionDistance(feature=2048, normalize=False)
        
        self.validation_encoded_latent = []

        if self.cfg['USE_SQRT_PENALTY']:
            self.aux = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def set_require_grad(self, nets, requires_grad=False):
        for param in nets.parameters():
            param.requires_grad = requires_grad

    def training_step(self, batch, batch_idx):
        x = batch
        batch_size = x.size(0)
        
        if self.cfg['USE_SQRT_PENALTY']:
            enc_opt, dec_opt, dis_opt, aux_opt = self.optimizers()
        else:
            enc_opt, dec_opt, dis_opt = self.optimizers()

        z_enc = self.encoder(x).detach()
        for _ in range(self.cfg['DISCRIMINATOR_STEPS']):
            dis_opt.zero_grad()
            z_prior = self.prior.sample(batch_size, self.device)
            discriminator_loss = -(self.target_transform(self.discriminator(z_enc))+self.prior_transform(self.discriminator(z_prior))).mean()
            self.manual_backward(discriminator_loss)
            self.clip_gradients(dis_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            dis_opt.step()
        
        self.set_require_grad(self.discriminator, False)
        z_enc = self.encoder(x)
        z_prior = self.prior.sample(batch_size, self.device)
        raw_penalty = (self.target_transform(self.discriminator(z_enc))+self.prior_transform(self.discriminator(z_prior))).mean()
        penalty_term = raw_penalty
        if self.cfg['USE_SQRT_PENALTY']:
            for _ in range(self.cfg['AUX_STEPS']):
                aux_opt.zero_grad()
                self.manual_backward(raw_penalty.detach() / torch.exp(self.aux) + torch.exp(self.aux) / 4.0)
                self.clip_gradients(aux_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                aux_opt.step()
            penalty_term = raw_penalty / torch.exp(self.aux.detach()) + torch.exp(self.aux.detach()) / 4.0

        x_recon = self.decoder(z_enc)
        recon_loss = nn.MSELoss(reduction='sum')(x_recon, x) / batch_size
        tot_loss = recon_loss + self.cfg['LAMBDA'] * penalty_term
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        self.manual_backward(tot_loss)
        self.clip_gradients(enc_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(dec_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        enc_opt.step()
        dec_opt.step()
        self.set_require_grad(self.discriminator, True)

        self.log_dict({
            "train/tot_loss": tot_loss,
            "train/recon_loss": recon_loss,
            "train/penalty_term": penalty_term,
        }, prog_bar=True)

        if self.cfg['USE_SQRT_PENALTY']:
            self.log("train/raw_penalty", raw_penalty, prog_bar=False)
            self.log("train/aux", self.aux.item(), prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x = batch
        batch_size = x.size(0)
        z_enc = self.encoder(x)
        x_recon = self.decoder(z_enc)

        self.validation_encoded_latent.append(z_enc.cpu())

        recon_loss = nn.MSELoss(reduction='sum')(x_recon, x) / batch_size
        self.log("val/recon_loss", recon_loss, sync_dist=True, prog_bar=True)

        if batch_idx == 0 and self.global_rank == 0:
            n = min(x.size(0), 16)
            grid = torchvision.utils.make_grid(
                torch.cat([x[:n], x_recon[:n]]),
                nrow=n//2, normalize=True, value_range=(-1, 1)
            )
            if self.logger:
                self.logger.experiment.add_image('val/real_vs_recon', grid, self.current_epoch)

        real_imgs = ((x * 0.5 + 0.5) * 255).to(torch.uint8)
        fake_imgs = ((self.decoder(self.prior.sample(batch_size, self.device)) * 0.5 + 0.5) * 255).to(torch.uint8)
        
        self.fid.update(real_imgs, real=True)
        self.fid.update(fake_imgs, real=False)

    def on_validation_epoch_end(self):
        fid_score = self.fid.compute()
        self.log("val_fid", fid_score, sync_dist=True, prog_bar=True)
        self.fid.reset()

        local_z_enc = torch.cat(self.validation_encoded_latent, dim=0).to(self.device)
        global_z_enc = self.all_gather(local_z_enc)

        if self.global_rank == 0:
            z_enc = global_z_enc.view(-1, self.cfg['LATENT_DIM'])
            z_prior = self.prior.sample(z_enc.size(0), self.device)

            print(f">>> Computing Latent PRDC (Dim={self.cfg['LATENT_DIM']}) with {z_enc.size(0)} samples...")
            metrics = compute_prdc(z_enc.float().cpu().numpy(), z_prior.float().cpu().numpy(), nearest_k=5)                
            self.log("val/precision", metrics['precision'].item(), rank_zero_only=True, prog_bar=False)
            self.log("val/recall", metrics['recall'].item(), rank_zero_only=True, prog_bar=False)
            self.log("val/density", metrics['density'].item(), rank_zero_only=True, prog_bar=False)
            self.log("val/coverage", metrics['coverage'].item(), rank_zero_only=True, prog_bar=False)
            
            gen_imgs = self.decoder(self.prior.sample(64, self.device))
            grid = torchvision.utils.make_grid(gen_imgs, nrow=8, normalize=True, value_range=(-1, 1))
            if self.logger:
                self.logger.experiment.add_image('Random_Generation', grid, self.current_epoch)
        
        self.validation_encoded_latent = []

    def on_train_epoch_end(self):
        for s in self.lr_schedulers():
            s.step()

    def configure_optimizers(self):
        enc_opt = optim.AdamW(self.encoder.parameters(), lr=self.cfg['LR'], betas=(0.5, 0.999))
        dec_opt = optim.AdamW(self.decoder.parameters(), lr=self.cfg['LR'], betas=(0.5, 0.999))
        dis_opt = optim.Adam(self.discriminator.parameters(), lr=self.cfg['LR'] * 2.5, betas=(0.5, 0.999))
        
        optimizers = [enc_opt, dec_opt, dis_opt]
        
        if self.cfg['USE_SQRT_PENALTY']:
            aux_opt = optim.Adam([self.aux], lr=self.cfg['LR'] * 10, betas=(0.5, 0.999))
            optimizers.append(aux_opt)
        
        schedulers = []
        max_epochs = self.cfg['EPOCHS']

        schedulers.append({"scheduler": optim.lr_scheduler.CosineAnnealingLR(enc_opt, T_max=max_epochs, eta_min=1e-6), "interval": "epoch", "frequency": 1})
        schedulers.append({"scheduler": optim.lr_scheduler.CosineAnnealingLR(dec_opt, T_max=max_epochs, eta_min=1e-6), "interval": "epoch", "frequency": 1})
        schedulers.append({"scheduler": optim.lr_scheduler.CosineAnnealingLR(dis_opt, T_max=max_epochs, eta_min=1e-6), "interval": "epoch", "frequency": 1})

        if self.cfg['USE_SQRT_PENALTY']:
            schedulers.append({"scheduler": optim.lr_scheduler.CosineAnnealingLR(aux_opt, T_max=max_epochs, eta_min=1e-6), "interval": "epoch", "frequency": 1})
            
        return optimizers, schedulers