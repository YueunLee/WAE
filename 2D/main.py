import argparse, datetime
import pytz
from collections import deque
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch
from torch import nn, optim
from architecture import MLPBlock, DenseICNN, RealNVP
from utils import init_distributions, wasserstein_distance, plot_losses
from mmd import mmd_batched_equivalence_test
from pathlib import Path


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argparse.ArgumentParser()
    
    # global settings
    args.add_argument("--learning-rate", type=float, default=1e-5)
    args.add_argument("--hidden-dim", type=int, default=128)
    args.add_argument("--div-hidden-dim", type=int, default=128)
    args.add_argument("--pot-layers", type=int, default=2)
    args.add_argument("--dec-layers", type=int, default=3)
    args.add_argument("--div-layers", type=int, default=8)
    args.add_argument("--seed", type=int, default=40, help="distribution seed")
    args.add_argument("--init-seed", type=int, default=2026, help="encoder initialization seed")
    
    # training settings
    args.add_argument("--penalty", type=str, default="fgan_js", 
        help="Penalty function to use. Options are: \
        'fgan_js', 'fgan_sqHellinger', 'sqrt_fgan_js', 'sqrt_fgan_sqHellinger', \
        'wgan', 'mmd'. Default is 'fgan_js'."
    )
    args.add_argument("--penalty-coef", type=float, default=20.0)
    args.add_argument("--batch-size", type=int, default=256)
    args.add_argument("--epochs", type=int, default=500)
    args.add_argument("--iter-per-epoch", type=int, default=150)
    args.add_argument("--adv-steps", type=int, default=10)
    args.add_argument("--aux-steps", type=int, default=5)
    args.add_argument("--scheduling", type=bool, default=True)
    args.add_argument("--anneal", type=float, default=1.0)
    args.add_argument("--train-seed", type=int, default=2)
    args.add_argument("--current", type=str, default=None, help="Custom result directory path")
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
    
    # Compute true Wasserstein distance for reference
    true_wasserstein = wasserstein_distance(mu=data_dist, potential=potential.push)
    
    # q = h o T*; h has same structure with the decoder
    enc_optim = optim.RAdam(encoder_net.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    encoder = lambda x: encoder_net(potential.push(x.requires_grad_()))
    
    div_optim = optim.RAdam(divergence.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    
    if args.scheduling:
        enc_sched = optim.lr_scheduler.StepLR(enc_optim, 10, gamma=0.9)
        div_sched = optim.lr_scheduler.StepLR(div_optim, 10, gamma=0.9)

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
    if args.current is not None:
        figpath = args.current + "/"
    else:
        figpath = current.strftime("%y%m%d_%H%M%S/")
    figpath += f"trainseed={str(train_seed)}/{penalty_name}"  

    adversarial = "gan" in penalty_name
    if "sqrt" in penalty_name:
        aux = torch.tensor([0.0], requires_grad=True)
        penalty_name = penalty_name[5:]
        aux_optim = optim.RAdam([aux], lr=0.0002)
        if scheduling:
            aux_sched = optim.lr_scheduler.StepLR(aux_optim, 10, gamma=0.9)
    else:
        aux = None

    penalty_func = getattr(__import__("penalties"), penalty_name+"_penalty")

    arr_recon, arr_penalty, arr_obj, arr_aux, arr_neg_pen = [], [], [], [], []
    arr_equiv_stat, arr_equiv_ubd, arr_equivalence = [], [], []

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
    test_sample_size = 10000
    test_block_size = int(sqrt(test_sample_size))
    test_interval = 10
    pass_rate_deque = deque(maxlen=5)  # Initialize deque for sliding window of pass rates
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

        if (epoch + 1) % test_interval == 0:
            encoder_net.eval()
            data = data_dist.rsample((test_sample_size, )).to(device)
            z_encoded = encoder(data)
            z = prior_dist.rsample((test_sample_size, )).to(device)
            ### MMD Batched Equivalence Test
            mmd_result = mmd_batched_equivalence_test(
                z_encoded.detach(),
                z.detach(),
                block_size=test_block_size,
                delta=0.00081, # delta = mean(MMD^2) + 2*std(MMD^2) from 1000 tests with optimal encoder
                alpha=0.05 / (epochs // test_interval),
                multiscale=True,
            )

            arr_equiv_stat.append(mmd_result["mmd_mean"])
            arr_equiv_ubd.append(mmd_result["upper_bound"])
            arr_equivalence.append(mmd_result["is_equivalent"])
            pass_rate_deque.append(mmd_result["is_equivalent"])  # Add to deque
            print(
                f"[{(epoch+1):04d}] obj: {arr_obj[-1]:<10.5g}" + f" recon: {arr_recon[-1]:<10.5g} penalty: {arr_penalty[-1]:<10.5g}" \
                + f"\nEquivalence-test: statistic = {arr_equiv_stat[-1]:<10.5g} Upper_bound = {arr_equiv_ubd[-1]:<10.5g} equivalence = {arr_equivalence[-1]:<10.5g}"
            )

            # Check if 3 or more out of the last (up to) 5 tests passed
            if len(pass_rate_deque) >= 3 and sum(pass_rate_deque) >= 3:
                print(f"[Completed in epoch {epoch+1}] Passed {sum(pass_rate_deque)} out of last {len(pass_rate_deque)} tests")
                break
        else:
            print(f"[{(epoch+1):04d}] obj: {arr_obj[-1]:<10.5g}" + f" recon: {arr_recon[-1]:<10.5g} penalty: {arr_penalty[-1]:<10.5g}")
            arr_equiv_stat.append(0.0)
            arr_equiv_ubd.append(0.0)
            arr_equivalence.append(-1)

        if scheduling:
            enc_sched.step()
            if adversarial:
                div_sched.step()
            if aux is not None:
                aux_sched.step()

        if anneal is not None:
            penalty_coef += anneal

    encoder_net.eval()    
    data = data_dist.rsample((test_sample_size, )).to(device)
    z_encoded = encoder(data)
    z = prior_dist.rsample((test_sample_size, )).to(device)
    mmd_result = mmd_batched_equivalence_test(
        z_encoded.detach(),
        z.detach(),
        block_size=test_block_size,
        delta=0.00081,
        alpha=0.05 / (epochs // test_interval),
        multiscale=True,
    )

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

    fig_title = f"$\lambda$: {penalty_coef} / MMD Upper Bound: {mmd_result['upper_bound']:<10.5g}"
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
        file.write("epoch | obj        | recon      | penalty    | equiv_stat | equiv_ubd  | equivalence | neg_pen    \n")
        for i in range(len(arr_obj)):
            file.write(
                f"{(i+1):>5d} | {arr_obj[i]:<10.5g} | {arr_recon[i]:<10.5g} | {arr_penalty[i]:<10.5g} | " \
                + f"{arr_equiv_stat[i]:<10.5g} | {arr_equiv_ubd[i]:<10.5g} | {arr_equivalence[i]:<10.3g} | {arr_neg_pen[i]}\n"
            )
        file.write(f"[MMD test] mmd_stat: {mmd_result['mmd_mean']:<10.5g}\tmmd_ubd: {mmd_result['upper_bound']:<10.5g}\tequivalence: {mmd_result['is_equivalent']}\n")

    ### store the state of trained encoder
    torch.save(encoder_net.state_dict(), figpath + "_model.pt")

    ### Plot comparison of z_encoded and z distributions
    n_samples = 10000
    data_samples = data_dist.rsample((n_samples,)).to(device)
    z_encoded_samples = encoder(data_samples).detach().cpu().numpy()
    z_prior_samples = prior_dist.rsample((n_samples,)).detach().cpu().numpy()

    '''
        2D scatter plot
    '''
    # plt.figure(figsize=(8, 6))
    # plt.scatter(z_encoded_samples[:, 0], z_encoded_samples[:, 1], alpha=0.5, label='Encoded (z_encoded)', color='blue', s=5)
    # plt.scatter(z_prior_samples[:, 0], z_prior_samples[:, 1], alpha=0.5, label='Prior (z)', color='red', s=5)
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.title('Comparison of Encoded and Prior Distributions')
    # plt.legend()
    # plt.axis('equal')
    # plt.savefig(figpath + "_distributions.png")
    # plt.close()

    '''
        2D density contour plot
    '''
    plt.figure(figsize=(14, 4))
    plt.subplot(131)
    sns.kdeplot(
        x=z_encoded_samples[:, 0], 
        y=z_encoded_samples[:, 1], 
        fill=False,
        color='blue',
        linewidths=2,
        levels=7,
        label='Encoded (Q_Z)'
    )
    sns.kdeplot(
        x=z_prior_samples[:, 0], 
        y=z_prior_samples[:, 1], 
        fill=False,
        color='red',
        linewidths=2,
        levels=7,
        label='Prior (P_Z)'
    )
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Density contour plots of Encoded and Prior Distributions")
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(132)
    arr_equiv_stat = np.array(arr_equiv_stat)
    epochs = np.arange(1, len(arr_equiv_stat)+1)
    plt.plot(epochs[arr_equiv_stat != 0], arr_equiv_stat[arr_equiv_stat != 0], marker='o', markersize=3)
    plt.title("Equivalence test statistic")
    plt.xlabel("epoch")

    plt.subplot(133)
    arr_equiv_ubd = np.array(arr_equiv_ubd)
    epochs = np.arange(1, len(arr_equiv_ubd)+1)
    plt.plot(epochs[arr_equiv_ubd != 0], arr_equiv_ubd[arr_equiv_ubd != 0], marker='o', markersize=3)
    plt.axhline(0.00081, color='r')
    plt.title("Equivalence test upper bound")
    plt.xlabel("epoch")

    plt.savefig(figpath + "_distributions.png")
    plt.close()
