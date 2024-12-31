import torch
from torch import nn, optim
from torch.distributions import uniform

import numpy as np

from pathlib import Path

from copy import deepcopy
from utils import init_distributions, init_decoder, wasserstein_distance, ks_test, plot_losses, plot_encoder, plot_auxiliary, save_data

class MLPBlock(nn.Module):
    def __init__(self, 
                input_dim: int,
                output_dim: int,
                hidden_dim: int, 
                num_layers: int,
                batchnorm: bool,
                last_activation: bool,
                activation: nn.Module = nn.Sigmoid,
                seed: int=0):
        
        super().__init__()
        encoder_list = []
        
        assert num_layers >= 1
        if num_layers == 1:
            encoder_list.append(
                nn.Linear(input_dim, output_dim)
            )
        else: # num_layers >= 2
            encoder_list.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batchnorm else nn.Identity(),
                activation(),
            ])
            for _ in range(num_layers-2):
                encoder_list.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim) if batchnorm else nn.Identity(),
                    activation(),
                ])
            encoder_list.extend([
                nn.Linear(hidden_dim, output_dim),
            ])
        
        if last_activation:
            encoder_list.append(activation())
            
        self.model = nn.Sequential(*encoder_list)
        self.seed = seed
        self.reset()

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def reset(self, train_seed:int=None):
        if train_seed is None:
            train_seed = self.seed
        torch.manual_seed(train_seed)
        for layer in self.model:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

class session():

    def __init__(self, 
                device: torch.device,
                dist_family: str,
                learning_rate: float,
                hidden_dim: int,
                enc_num_layers: int,
                div_num_layers: int,
                seed: int,
                init_seed: int,
                current: str,
                scaleX: float,
                scaleG: float,
                coefX: float,
                coefG: float):
        
        ### seed, device setting        
        self.device = device
        self.seed = seed
        self.lr = learning_rate
        torch.manual_seed(self.seed)

        ### distributions initialization
        self.data_dist, self.gen_dist, PG_scale, PG_coef = init_distributions(dist_family, scaleX, scaleG, coefX, coefG)
        self.prior_dist = uniform.Uniform(low = 0.0, high = 1.0)

        ### true wasserstein distance
        self.true_wasserstein = wasserstein_distance(self.data_dist, self.gen_dist, dist_family)
        print(f"W(P_X, P_G) = {self.true_wasserstein.item():.6f}")
        
        ### decoder & encoder & divergence
        self.decoder = lambda z: init_decoder(z, self.gen_dist)
        self.encoder = MLPBlock(
            input_dim=1,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=enc_num_layers,
            batchnorm=False,
            last_activation=True,
            activation=nn.Sigmoid,
            seed=init_seed
        ).to(self.device)        
        self.divergence = MLPBlock(
            input_dim=1,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=div_num_layers,
            batchnorm=False,
            last_activation=False,
            seed=init_seed
        ).to(self.device)    

        self.current = current

        # lambda_threshold = 2 * diam(X) * max{1, Lipschitz constant of g} * (divergence dependent constant)
        # Lipschitz constant of g = scale * sqrt(2 * pi) * exp(coef^2 * 0.5)
        # Divergence dependent constant = 1/sqrt(f"(1)) = 1 for KL, reverse-KL, WGAN, 1/sqrt(2) for pearson, neyman, sqrt(2) for JS, sqHellinger
        PX_diam = ((self.data_dist.b - self.data_dist.a) * self.data_dist.scale).item() # diam (X)
        print(f"diam(P_X)={PX_diam:<10.5g}\tP_G-scale={PG_scale:<10.5g}\tP_G-coef={PG_coef:<10.5g}")
        self.lamb_threshold = 2.0 * PX_diam * (PG_scale * np.sqrt(np.pi * 2.0) * np.exp(PG_coef**2 * 0.5))

        self.encoder_weights_stored = deepcopy(self.encoder.state_dict())
        self.divergence_weights_stored = deepcopy(self.divergence.state_dict())
    
    def train(self, args):
        
        # get attributes
        penalty_name=args.penalty
        penalty_coef=args.penalty_coef
        batch_size=args.batch_size
        epochs=args.epochs
        iter_per_epoch=args.iter_per_epoch
        adv_steps=args.adv_steps
        aux_steps=args.aux_steps
        scheduling=args.scheduling
        use_threshold=args.use_threshold
        anneal=args.anneal
        train_seed=args.train_seed

        ### supplemental settings
        adversarial = "gan" in penalty_name
        if "sqrt" in penalty_name:
            aux = torch.tensor([0.0], requires_grad=True)
            penalty_name = penalty_name[5:]
        else:
            aux = None
            
        if use_threshold == True:
            penalty_coef = self.lamb_threshold
            if penalty_name in ["fgan_kl", "sqrt_fgan_kl", "fgan_reverse_kl", "sqrt_fgan_reverse_kl", "wgan", "mmd"]:
                penalty_coef *= 1
            elif penalty_name in ["fgan_pearson", "sqrt_fgan_pearson", "fgan_neyman", "sqrt_fgan_neyman"]:
                penalty_coef *= 1 / np.sqrt(2)
            elif penalty_name in ["fgan_js", "sqrt_fgan_js", "fgan_sqHellinger", "sqrt_fgan_sqHellinger"]:
                penalty_coef *= np.sqrt(2)
            else:
                raise ValueError(f"Unknown penalty_name: {penalty_name}")
            penalty_coef = int(penalty_coef / 10.0 + 1.0) * 10.0 # ceiling
            args.penalty_coef = penalty_coef
            
        print(f"penalty_coef: {penalty_coef:<10.5g}")
        
        ### initializing encoder and divergence with stored weights)
        self.encoder.load_state_dict(self.encoder_weights_stored)
        self.divergence.load_state_dict(self.divergence_weights_stored)

        ### optimizer & scheduler setting
        enc_optim = optim.RAdam(self.encoder.parameters(), lr=self.lr, betas=(0.5, 0.999))
        div_optim = optim.RAdam(self.divergence.parameters(), lr=self.lr, betas=(0.5, 0.999))
        if scheduling:
            enc_sched = optim.lr_scheduler.StepLR(enc_optim, 100, gamma=0.8)
            div_sched = optim.lr_scheduler.StepLR(div_optim, 100, gamma=0.8)
            
        if aux is not None:
            aux_optim = optim.RAdam([aux], lr=0.001)
            aux_sched = optim.lr_scheduler.StepLR(aux_optim, 100, gamma=0.8)

        torch.manual_seed(train_seed)
        
        ### penalty function setting
        penalty_func = getattr(__import__("penalties"), penalty_name+"_penalty")

        ### auxiliary variable initialization
        if aux is not None:
            self.encoder.eval()
            first_x = self.data_dist.rsample((batch_size, 1)).to(self.device)
            first_z = self.encoder(first_x)
            penalty = -penalty_func(
                z_hat=first_z, 
                prior_dist=self.prior_dist,
                discriminator=self.divergence, 
                adversarial=True,
            ).detach().cpu() 
            if penalty > 0.0:
                aux.data = .5 * (4.0 * penalty).log()

        ### arrays for recording
        arr_recon, arr_penalty, arr_obj, arr_aux, arr_neg_pen = [], [], [], [], []
        arr_ks_stat, arr_ks_pval, arr_pass_rate = [], [], []

        ### training
        epoch = 1
        final_lambda = penalty_coef
        tmp = True
        ks_pass_cnt = 0
        while epoch <= epochs:
            total_recon, total_penalty, total_aux = 0.0, 0.0, 0.0
            neg_pen_cnt = 0
            self.encoder.train()
            self.divergence.train()
            for _ in range(iter_per_epoch):
                enc_optim.zero_grad()
                if adversarial:
                    for _ in range(adv_steps):
                        data = self.data_dist.rsample((batch_size,1)).to(self.device)
                        z_encoded = self.encoder(data)
                        div_optim.zero_grad()
                        div_penalty = penalty_func(
                                z_hat=z_encoded.detach(),
                                prior_dist=self.prior_dist,
                                discriminator=self.divergence,
                                adversarial=True,
                            )
                        div_penalty.backward()
                        div_optim.step()
                else: 
                    data = self.data_dist.rsample((batch_size,1)).to(self.device)
                    z_encoded = self.encoder(data)

                if aux is not None:
                    penalty = -penalty_func(
                            z_hat=z_encoded,
                            prior_dist=self.prior_dist,
                            discriminator=self.divergence,
                            adversarial=True
                        ).detach().cpu()
                    if penalty < 0.0:
                        neg_pen_cnt += 1
                    for _ in range(aux_steps-1):
                        aux_optim.zero_grad()
                        data = self.data_dist.rsample((batch_size,1)).to(self.device)
                        z_encoded = self.encoder(data)
                        penalty_temp = -penalty_func(
                                z_hat=z_encoded.detach(),
                                prior_dist=self.prior_dist,
                                discriminator=self.divergence,
                                adversarial=True
                            ).detach().cpu()
                        sqrt_penalty_temp = penalty_temp * (-aux).exp() + aux.exp() * 0.25
                        sqrt_penalty_temp.backward()
                        aux_optim.step()
                    aux_optim.zero_grad()
                    sqrt_penalty = penalty * (-aux).exp() + aux.exp() * 0.25
                    sqrt_penalty.backward()
                    aux_optim.step()

                penalty = penalty_func(
                    z_hat=z_encoded,
                    prior_dist=self.prior_dist,
                    discriminator=self.divergence,
                    adversarial=False,
                )

                if aux is not None:
                    penalty *= (-aux).exp().item()
                
                x_recon = self.decoder(z_encoded)
                recon_loss = (data - x_recon).pow(2).mean() # L2 loss
                loss = recon_loss + penalty_coef * penalty
                loss.backward()
                enc_optim.step()

                penalty = -penalty_func(
                    z_hat=z_encoded.detach(),
                    prior_dist=self.prior_dist,
                    discriminator=self.divergence,
                    adversarial=True,
                    lambda_gp=0.0
                ).detach()

                if not adversarial:
                    penalty = -penalty

                if aux is not None:
                    penalty = penalty * (-aux).exp().item() + aux.exp().item() * 0.25
                    total_aux += aux.item()
                else:
                    if penalty.item() < 0.0:
                        neg_pen_cnt += 1

                total_recon += recon_loss.item()
                total_penalty += penalty.item()
            
            avg_recon = total_recon / iter_per_epoch
            avg_penalty = total_penalty / iter_per_epoch 
            arr_recon.append(avg_recon)
            arr_penalty.append(avg_penalty)
            arr_obj.append(avg_recon + penalty_coef * avg_penalty)
            arr_neg_pen.append(neg_pen_cnt)
            if aux is not None:
                avg_aux = total_aux / iter_per_epoch
                arr_aux.append(avg_aux)

            ks_stat, ks_pval, pass_rate = ks_test(self.encoder, self.data_dist, self.prior_dist, 6)
            arr_ks_stat.append(ks_stat)
            arr_ks_pval.append(ks_pval)
            arr_pass_rate.append(pass_rate)
            
            if epoch % 10 == 0:
                print(
                    f"[{epoch:04d}] train-obj: {arr_obj[-1]:<10.5g}" \
                    + f" train-recon: {arr_recon[-1]:<10.5g} train-penalty: {arr_penalty[-1]:<10.5g}" \
                    + f"\n       ks-test: statistic = {arr_ks_stat[-1]:<10.5g} p-value = {arr_ks_pval[-1]:<10.5g} pass_rate = {arr_pass_rate[-1]:<10.5g}"
                )

            epoch += 1
            if scheduling:
                enc_sched.step()
                if adversarial:
                    div_sched.step()
                if aux is not None:
                    aux_sched.step()

            if tmp and (arr_ks_pval[-1] > 1e-5):
                final_lambda = penalty_coef
                tmp = False

            if pass_rate >= 0.5:
                if ks_pass_cnt < 3:
                    ks_pass_cnt += 1
                    print(f"[Passed the KS test in epoch {epoch-1} with p-value {arr_ks_pval[-1]:<1.5g}]")
                else:
                    print(f"[Completed in epoch {epoch-1} with p-value {arr_ks_pval[-1]:<1.5g}]")
                    break
            
            if anneal is not None:
                penalty_coef += anneal
            
        epoch -= 1

        ### plot the losses and the encoder
        if aux is not None:
            penalty_name = "sqrt_" + penalty_name
        figpath = self.current
        figpath += f"trainseed={str(train_seed)}/{penalty_name}"  
        Path(figpath).mkdir(parents=True, exist_ok=True)
        if anneal is not None and anneal != 0.0:
            figpath += f"/linear"
        else:
            figpath += f"/none"
        
        with open(figpath + "_config.txt", "w") as file:
            file.write(
                ' '.join(f"{k}={v}\n" for k, v in vars(args).items())
            )

        self.encoder.eval()
        torch.manual_seed(0)
        total_recon = 0.0
        for _ in range(iter_per_epoch):
            data = self.data_dist.rsample((batch_size,1)).to(self.device)
            recon_loss = (data - self.decoder(self.encoder(data))).pow(2).mean()
            total_recon += recon_loss
        total_recon /= iter_per_epoch
        
        plot_losses(
            figpath=figpath,
            arr_recon=arr_recon, 
            arr_penalty=arr_penalty,
            arr_obj=arr_obj,
            true_val=self.true_wasserstein.item(),
            final_recon=total_recon
        )
        
        plot_encoder(
            figpath=figpath,
            encoder=self.encoder,
            data_dist=self.data_dist,
            prior_dist=self.prior_dist,
            arr_ks_stat=arr_ks_stat,
            arr_ks_pval=arr_ks_pval
        )

        ### plot the auxiliary variable
        if aux is not None:
            plot_auxiliary(
                figpath=figpath, 
                arr_aux=arr_aux
            )

        ### save the data as a file
        save_data(figpath=figpath,
                epoch=epoch,
                arr_obj=arr_obj,
                arr_recon=arr_recon, 
                arr_penalty=arr_penalty, 
                arr_ks_stat=arr_ks_stat, 
                arr_ks_pval=arr_ks_pval,
                arr_pass_rate=arr_pass_rate,
                arr_neg_pen=arr_neg_pen)

        ### store the state of trained encoder
        torch.save(self.encoder.state_dict(), figpath + "_model.pt")
        
        return final_lambda

