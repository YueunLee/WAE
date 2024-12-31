import argparse, datetime
import pytz

import torch
from torch import nn, optim
from architecture import MLPBlock, DenseICNN, RealNVP
from utils import init_distributions, wasserstein_distance, ks2d_test, plot_losses
from pathlib import Path


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argparse.ArgumentParser()
    
    # global settings
    args.add_argument("--learning-rate", type=float, default=0.0001)
    args.add_argument("--hidden-dim", type=int, default=128)
    args.add_argument("--div-hidden-dim", type=int, default=128)
    args.add_argument("--pot-layers", type=int, default=2)
    args.add_argument("--dec-layers", type=int, default=2)
    args.add_argument("--div-layers", type=int, default=8)
    args.add_argument("--seed", type=int, default=40, help="distribution seed")
    args.add_argument("--init-seed", type=int, default=2024, help="encoder initialization seed")
    
    # training settings
    args.add_argument("--penalty", type=str, default="fgan_js", 
        help="Penalty function to use. Options are: \
        'fgan_js', 'fgan_kl', 'fgan_reverse_kl', 'fgan_pearson', 'fgan_neyman', 'fgan_sqHellinger', \
        'sqrt_fgan_js', 'sqrt_fgan_kl', 'sqrt_fgan_reverse_kl', 'sqrt_fgan_pearson', 'sqrt_fgan_neyman', 'sqrt_fgan_sqHellinger', \
        'wgan', 'mmd'. Default is 'fgan_js'."
    )
    args.add_argument("--penalty-coef", type=float, default=100.0)
    args.add_argument("--batch-size", type=int, default=256)
    args.add_argument("--epochs", type=int, default=500)
    args.add_argument("--iter-per-epoch", type=int, default=150)
    args.add_argument("--adv-steps", type=int, default=20)
    args.add_argument("--aux-steps", type=int, default=10)
    args.add_argument("--scheduling", type=bool, default=True)
    args.add_argument("--anneal", type=float, default=2.0)
    args.add_argument("--train-seed", type=int, default=2)
    args = args.parse_args()

    current = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    
    torch.manual_seed(args.seed)

    # Brenier potential parameterized by DenseICNN
    potential = DenseICNN(
        dim=2,
        hidden_sizes=[args.hidden_dim] * args.pot_layers,
        weights_init_std=0.2
    )
    potential.convexify()

    # bijective decoder parameterized by RealNVP
    decoder = RealNVP(
        dim=2,
        num_layers=args.dec_layers,
        hidden_size=args.hidden_dim
    )

    potential = potential.to(device)
    potential.eval()
    decoder = decoder.to(device)
    decoder.eval()
    
    torch.manual_seed(args.init_seed)
    encoder_net = RealNVP(
        dim=2,
        num_layers=args.dec_layers,
        hidden_size=args.hidden_dim
    ) # optimal = inverse of the decoder

    divergence = MLPBlock(
        input_dim=2,
        output_dim=1,
        hidden_dim=args.div_hidden_dim,
        num_layers=args.div_layers,
        batchnorm=False,
        last_activation=False,
        activation=nn.LeakyReLU
    )
    encoder_net = encoder_net.to(device)
    divergence = divergence.to(device)

    # initialize distribution settings
    torch.manual_seed(args.seed)
    # q* = g^{-1} o T*; T*=optimal transport map
    optimal_encoder = lambda x: decoder(potential.push(x.requires_grad_()), True)
    
    data_dist, prior_dist = init_distributions(dim=2, optimal_encoder=optimal_encoder)
    
    true_wasserstein = wasserstein_distance(mu=data_dist, potential=potential.push)
    
    # q = h o T*; h has same structure with the decoder
    enc_optim = optim.RAdam(encoder_net.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    encoder = lambda x: encoder_net(potential.push(x.requires_grad_()))
    
    div_optim = optim.RAdam(divergence.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    
    if args.scheduling:
        enc_sched = optim.lr_scheduler.StepLR(enc_optim, 100, gamma=0.8)
        div_sched = optim.lr_scheduler.StepLR(div_optim, 100, gamma=0.8)

    # get attributes
    penalty_name=args.penalty
    penalty_coef=args.penalty_coef
    batch_size=args.batch_size
    epochs=args.epochs
    iter_per_epoch=args.iter_per_epoch
    adv_steps=args.adv_steps
    aux_steps=args.aux_steps
    scheduling=args.scheduling
    anneal=args.anneal
    train_seed=args.train_seed

    # save path
    figpath = current.strftime("%y%m%d/%H%M%S/")
    figpath += f"trainseed={str(train_seed)}/{penalty_name}"  

    adversarial = "gan" in penalty_name
    if "sqrt" in penalty_name:
        aux = torch.tensor([0.0], requires_grad=True)
        penalty_name = penalty_name[5:]
        aux_optim = optim.RAdam([aux], lr=0.0002)
        if scheduling:
            aux_sched = optim.lr_scheduler.StepLR(aux_optim, 100, gamma=0.8)
    else:
        aux = None

    penalty_func = getattr(__import__("penalties"), penalty_name+"_penalty")

    arr_recon, arr_penalty, arr_obj = [], [], []
    arr_aux, arr_neg_pen, arr_test_stat, arr_test_pval, arr_pass_rate = [], [], [], [], []

    ### auxiliary variable initialization
    if aux is not None:
        encoder_net.eval()
        first_x = data_dist.rsample((batch_size, )).to(device)
        first_z = encoder(first_x)
        penalty = -penalty_func(
            z_hat=first_z, 
            prior_dist=prior_dist,
            discriminator=divergence, 
            adversarial=True,
        ).detach().cpu()
        if penalty > 0.0:
            aux.data = .5 * (4.0 * penalty).log()

    if aux is not None: print(f"PENALTY : sqrt_{penalty_name}")
    else: print(f"PENALTY: {penalty_name}")

    torch.manual_seed(train_seed)
    test_pass_cnt = 0
    ks_test_epoch = 10
    pass_rate_threshold = 0.6
    num_iter_test = 3
    num_pass_test = 1
    for epoch in range(epochs):
        total_recon, total_penalty = 0.0, 0.0
        neg_pen_cnt = 0
        
        encoder_net.train() # train mode
        divergence.train()
        for _ in range(iter_per_epoch):
            enc_optim.zero_grad()
            if adversarial:
                for _ in range(adv_steps):
                    data = data_dist.rsample((batch_size, )).to(device)
                    z_encoded = encoder(data)
                    div_optim.zero_grad()
                    div_penalty = penalty_func(
                        z_hat=z_encoded.detach(),
                        prior_dist=prior_dist,
                        discriminator=divergence,
                        adversarial=True,
                    )
                    div_penalty.backward()
                    div_optim.step()
            else:
                data = data_dist.rsample((batch_size, )).to(device)
                z_encoded = encoder(data)

            if aux is not None:
                penalty = -penalty_func(
                    z_hat=z_encoded,
                    prior_dist=prior_dist,
                    discriminator=divergence,
                    adversarial=True
                ).detach().cpu()
                if penalty < 0.0:
                    neg_pen_cnt += 1
                for _ in range(aux_steps-1):
                    aux_optim.zero_grad()
                    data = data_dist.rsample((batch_size, )).to(device)
                    z_encoded = encoder(data)
                    penalty_temp = -penalty_func(
                            z_hat=z_encoded.detach(),
                            prior_dist=prior_dist,
                            discriminator=divergence,
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
                prior_dist=prior_dist,
                discriminator=divergence,
                adversarial=False,
            )

            if aux is not None:
                penalty *= (-aux).exp().item()
            
            x_recon = decoder(z_encoded)
            recon_loss = (data - x_recon).pow(2).sum(dim=1).mean() # L2 loss
            loss = recon_loss + penalty_coef * penalty
            loss.backward()
            enc_optim.step()

            penalty = -penalty_func(
                z_hat=z_encoded.detach(),
                prior_dist=prior_dist,
                discriminator=divergence,
                adversarial=True,
                lambda_gp=0.0
            ).detach()

            if not adversarial:
                penalty = -penalty

            if aux is not None:
                penalty = penalty * (-aux).exp().item() + aux.exp().item() * 0.25
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

        if (epoch + 1) % ks_test_epoch == 0:
            ks_stat, ks_pval, pass_rate = ks2d_test(encoder, data_dist, prior_dist, iter_test=num_iter_test, device=device)
            arr_test_stat.append(ks_stat)
            arr_test_pval.append(ks_pval)
            arr_pass_rate.append(pass_rate)
            print(
                f"[{(epoch+1):04d}] obj: {arr_obj[-1]:<10.5g}" + f" recon: {arr_recon[-1]:<10.5g} penalty: {arr_penalty[-1]:<10.5g}" \
                + f"\nks-test: statistic = {arr_test_stat[-1]:<10.5g} p-value = {arr_test_pval[-1]:<10.5g} pass_rate = {arr_pass_rate[-1]:<10.5g}"
            )

            # KS test is passed -> break
            if pass_rate >= pass_rate_threshold:
                if test_pass_cnt < (num_pass_test - 1):
                    test_pass_cnt += 1
                    print(f"[Passed the test in epoch {epoch+1} with p-value {arr_test_pval[-1]:<1.5g}]")    
                else:
                    print(f"[Completed in epoch {epoch+1} with p-value {arr_test_pval[-1]:<1.5g}]")
                    break
        else :
            print(f"[{(epoch+1):04d}] obj: {arr_obj[-1]:<10.5g}" + f" recon: {arr_recon[-1]:<10.5g} penalty: {arr_penalty[-1]:<10.5g}")
            arr_test_stat.append(0.0)
            arr_test_pval.append(0.0)
            arr_pass_rate.append(0.0)
        
        if scheduling:
            enc_sched.step()
            if adversarial:
                div_sched.step()
            if aux is not None:
                aux_sched.step()

        if anneal is not None:
            penalty_coef += anneal

    encoder_net.eval()    
    ### Kolmogorov-Smirnov test
    ks_stat, ks_pval, pass_rate = ks2d_test(encoder, data_dist, prior_dist, iter_test=5, device=device)
    print(f"KS-test: p-value = {ks_pval:<10.5g}\tpass_rate: {pass_rate:<10.5g}")

    Path(figpath).mkdir(parents=True, exist_ok=True)
    if anneal is not None and anneal != 0.0:
        figpath += f"/linear"
    else:
        figpath += f"/none"

    ### plot the losses and the encoder
    with open(figpath + "_config.txt", "w") as file:
        file.write(
            ' '.join(f"{k}={v}\n" for k, v in vars(args).items())
        )
    
    fig_title = f"$\lambda$: {penalty_coef} / KS: {ks_pval:<10.5g}"
    torch.manual_seed(0)
    total_recon = 0.0
    for _ in range(iter_per_epoch):
        data = data_dist.rsample((batch_size, )).to(device)
        recon_loss = (data - decoder(encoder(data))).pow(2).sum(dim=1).mean()
        total_recon += recon_loss
    total_recon /= iter_per_epoch
    plot_losses(
        figpath=figpath,
        arr_recon=arr_recon, 
        arr_penalty=arr_penalty,
        coef=penalty_coef,
        true_val=true_wasserstein.item(),
        fig_title=fig_title,
        final_recon=total_recon
    )

    ### save data
    with open(figpath+"_data.txt", "w") as file:
        file.write("epoch | obj        | recon      | penalty    | test_stat  | test_pval  | pass_rate   | neg_pen    \n")
        for i in range(len(arr_obj)):
            file.write(
                f"{(i+1):>5d} | {arr_obj[i]:<10.5g} | {arr_recon[i]:<10.5g} | {arr_penalty[i]:<10.5g} | " \
                + f"{arr_test_stat[i]:<10.5g} | {arr_test_pval[i]:<10.5g} | {arr_pass_rate[i]:<10.3g} | {arr_neg_pen[i]}\n"
            )
        file.write(f"[KS test] ks_stat: {ks_stat:<10.5g}\tks_pval: {ks_pval:<10.5g}\tpass_rate: {pass_rate}")

    ### store the state of trained encoder
    torch.save(encoder_net.state_dict(), figpath + "_model.pt")

