import os, argparse, importlib
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb

from utils import load_dset

def load_model(dataset: str, model: str, base_channel_size: int, latent_dim: int, 
               learning_rate: float, adv_learning_rate: float=0.0, sampler="normal"):
    encoder = getattr(importlib.import_module("networks"), "Encoder_"+dataset.lower())
    decoder = getattr(importlib.import_module("networks"), "Decoder_"+dataset.lower())
    wae = getattr(importlib.import_module("models"), model)
    
    num_input_channels = 1 if dataset.lower() == "mnist" else 3
    dict_param = {
        "num_input_channels": num_input_channels,
        "base_channel_size": base_channel_size,
        "latent_dim": latent_dim,
        "learning_rate": learning_rate,
        "encoder": encoder,
        "decoder": decoder,
        "sampler": sampler, # z-sampler
    }
    if model in ["WAE_GAN", "WAE_KL"]:
        discriminator = getattr(importlib.import_module("networks"), "Discriminator_"+dataset.lower())
        dict_param.update({"adv_learning_rate": adv_learning_rate, "discriminator": discriminator})
    return wae(**dict_param)


def main(args):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="wae_penalty_method",
        entity="hjlee22"
    )
    wandb.define_metric("gen_fid", summary="min")
    wandb.config.update(args)

    # Setting
    torch.manual_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets & dataloaders
    train_dset, val_dset = load_dset(args.dataset, args.data_dir)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = load_model(args.dataset, args.model, args.base_channel_size, args.latent_dim,
                       args.learning_rate, args.adv_learning_rate, args.sampler)
    model = model.to(device)
    
    # optimizers
    dict_optim = model.get_optimizers()
    optimizer = dict_optim["optimizer"][0]
    scheduler = dict_optim["scheduler"][0]
    
    adv_optimizer, adv_scheduler = None, None
    if len(dict_optim["optimizer"]) > 1:
        adv_optimizer = dict_optim["optimizer"][-1]
        adv_scheduler = dict_optim["scheduler"][-1]

    # Annealing lambda; Penalty methods
    anneal_func = getattr(importlib.import_module("utils"), args.penalty_anneal.lower()+"_anneal")

    # train & validation
    dict_lamb = {
        "recon_loss": 1.0, 
        "mmd_penalty": args.lambda_mmd, 
        "gan_penalty": args.lambda_gan,
        "adv_loss": 1.0,
    }
    
    # Sample latent vectors
    if args.sampler.lower() == "uniform":
        z_smpl = torch.rand(64, args.latent_dim).to(device)
        z_smpl = (z_smpl - .5) / .5
    else:
        z_smpl = torch.randn(64, args.latent_dim).to(device)
    x_smpl = next(iter(val_loader))[0].to(device) # first batch
    
    # initialize auxiliary variable to compute the square root of the JS divergence
    # aux = 0.0 if args.exact else None
    aux = None
    if args.exact:
        aux = torch.tensor([0.0], requires_grad=args.aux_sgd)
        if args.aux_sgd:
            aux_optim = optim.SGD([aux], lr=0.001)
            # aux_sched = optim.lr_scheduler.StepLR(aux_optim, step_size=10, gamma=0.8)

    ### Training & Evaluating Models
    fid = FrechetInceptionDistance(normalize=True).to(device)
    best_genfid = float("inf")
    model = torch.compile(model) # pytorch 2.0: speed up
    for epoch in range(args.epochs):
        # train
        tqdm_train_loader = tqdm(train_loader)
        model.train()
        total_loss = defaultdict(float)
        pen_neg_cnt = 0

        # Annealing dict_lambda; Penalty methods
        dict_lamb.update({
            "mmd_penalty": anneal_func(epoch, args.lambda_mmd, args.epochs),
            "gan_penalty": anneal_func(epoch, args.lambda_gan, args.epochs)
        })
        
        for i, (data, _) in enumerate(tqdm_train_loader):
            tqdm_train_loader.set_description(f"Train epoch {epoch}")

            ingredient = model.first_operation(data.to(device))
            if adv_optimizer is not None:
                # train discriminator ("adv_steps" times)
                adv_optimizer.zero_grad()
                dict_adv_loss = model.get_adv_losses(ingredient)
                loss = sum([dict_adv_loss[key] for key in dict_adv_loss.keys()])
                loss.backward()
                adv_optimizer.step()

            if aux is not None:
                penalty2 = -model.get_adv_losses(ingredient)["adv_loss"].item()
                if "GAN" in args.model:
                    penalty2 += np.log(4)

                if penalty2 < 0.0:
                    pen_neg_cnt += 1
                if not args.aux_sgd:
                    if penalty2 > 0.0:
                        for _ in range(args.aux_steps):
                            aux -= (-penalty2 * (-2.0 * aux).exp() + .25) / (penalty2 * (-2.0 * aux).exp() + .25)
                    else:
                        aux = 0.0
                else:
                    aux_optim.zero_grad()
                    sqrt_penalty = penalty2 * (-aux).exp() + aux.exp() * 0.25
                    sqrt_penalty.backward()
                    aux_optim.step()

                total_loss["aux"] += aux

            # train encoder, decoder
            optimizer.zero_grad()
            dict_loss = model.get_losses(ingredient, aux=aux)
            loss = sum([dict_lamb[key] * dict_loss[key] for key in dict_loss.keys()])
            loss.backward()
            optimizer.step()
            
            for key in dict_loss.keys():
                total_loss[key] += dict_loss[key]

            tqdm_train_loader.set_postfix(loss=loss.item())
        
        
        wandb.log({'train_'+k: v / len(train_loader) for k, v in total_loss.items()}, step=(epoch+1))
        wandb.log({'pen_neg_cnt': pen_neg_cnt / len(train_loader)}, step=(epoch+1))
        if scheduler is not None:
            scheduler.step()
        if adv_scheduler is not None:
            adv_scheduler.step()
    
        # validation
        tqdm_val_loader = tqdm(val_loader)
        model.eval()
        total_val_loss = {k: 0.0 for k in total_loss.keys()}
        if aux is not None:
            del total_val_loss["aux"]
        with torch.no_grad():
            for i, (data, _) in enumerate(tqdm_val_loader):
                tqdm_val_loader.set_description(f"Val epoch {epoch}")
                ingredient = model.first_operation(data.to(device))
                dict_val_loss = model.get_losses(ingredient)
                val_loss = sum([dict_lamb[key] * dict_val_loss[key] for key in dict_val_loss.keys()])
                
                for key in total_val_loss.keys():
                    total_val_loss[key] += dict_val_loss[key]

                # converting input images into 3 channels
                x = data if data.size(1) == 3 else torch.cat([data]*3, dim=1)
                x = x.to(device)
                fid.update((x+1.0)*0.5, real=True) # value range: [-1, 1] -> [0, 1]
                
                if args.sampler.lower() == "uniform":
                    z_prior = torch.rand(data.size(0), args.latent_dim).to(device)
                    z_prior = (z_prior - .5) / .5
                elif args.sampler.lower() == "normal":
                    z_prior = torch.randn(data.size(0), args.latent_dim).to(device)
                x_gen = (model.decode(z_prior)+1.0)*0.5 # value range: [-1, 1] -> [0, 1]
                
                x_gen_fid = x_gen if x_gen.size(1) == 3 else torch.cat([x_gen]*3, dim=1)
                fid.update(x_gen_fid, real=False)
                
                tqdm_val_loader.set_postfix(loss=val_loss.item())

            wandb.log({'val_'+k: v / len(val_loader) for k, v in total_val_loss.items()}, step=(epoch+1))

            # FID score
            genfid = fid.compute()
            wandb.log({'gen_fid': fid.compute()}, step=(epoch+1))
            if genfid < best_genfid:
                wandb.run.summary["best_gen_fid"] = genfid
                best_genfid = genfid
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best.pt"))

            # reconstruction
            recon = model(x_smpl[:32])
            x_recon = torch.cat((x_smpl[:32], recon), dim = 0).cpu()
            grid = make_grid((x_recon+1.0)*0.5)
            
            # Sample generation
            gen_img = model.decode(z_smpl).cpu()
            gen_grid = make_grid((gen_img+1.0)*0.5)
            wandb.log({"recon": wandb.Image(grid), "generation": wandb.Image(gen_grid)}, step=(epoch+1))

        fid.reset()
    
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "last.pt"))
    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="WAE Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="CelebA")
    parser.add_argument("--data-dir", type=str, default="./data", help="path to training image.")
    
    parser.add_argument("--model", type=str, default="WAE_GAN", help="choose among 2 models: WAE_MMD, WAE_GAN")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="learning rate for encoder-decoder optimizer.")
    parser.add_argument("--adv-learning-rate", type=float, default=1e-3, help="learning rate for adversarial optimizer.")
    parser.add_argument("--aux-steps", type=int, default=10, help="steps for updating auxiliary variable.")
    parser.add_argument("--aux-sgd", type=bool, default=False, help="Updating auxiliary variable by SGD or Newton-Raphson")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-channel-size", type=int, default=128)

    parser.add_argument("--sampler", type=str, default="uniform", help="z-samplers: normal, uniform")
    parser.add_argument("--lambda-mmd", type=float, default=100.0)
    parser.add_argument("--lambda-gan", type=float, default=1.0)
    parser.add_argument("--penalty-anneal", type=str, default="none", help="choose among 3 types: none, linear, exponential")
    parser.add_argument("--exact", type=bool, default=False, help="use exact penalty (sqrt of a f-divergence) or not")
    
    parser.add_argument("--random-seed", type=int, default=2024, help="random seed for reproducibility.")
    args = parser.parse_args()
    
    main(args)