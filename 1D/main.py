import argparse

import torch
from structure import session
import datetime

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist_family = "truncated_normal"
    args = argparse.ArgumentParser()
    
    ### global settings
    args.add_argument("--learning-rate", type=float, default=0.0005)
    args.add_argument("--hidden-dim", type=int, default=32)
    args.add_argument("--enc-layers", type=int, default=2)
    args.add_argument("--div-layers", type=int, default=3)
    args.add_argument("--seed", type=int, default=1000, help="distribution seed")
    args.add_argument("--init-seed", type=int, default=2024, help="encoder initialization seed")
    
    # training settings
    args.add_argument("--penalty", type=str, default="fgan_js", 
        help="Penalty function to use. Options are: \
        'fgan_js', 'fgan_kl', 'fgan_reverse_kl', 'fgan_pearson', 'fgan_neyman', 'fgan_sqHellinger', \
        'sqrt_fgan_js', 'sqrt_fgan_kl', 'sqrt_fgan_reverse_kl', 'sqrt_fgan_pearson', 'sqrt_fgan_neyman', 'sqrt_fgan_sqHellinger', \
        'wgan', 'mmd'. Default is 'fgan_js'."
    )
    args.add_argument("--penalty-coef", type=float, default=20.0)
    args.add_argument("--batch-size", type=int, default=256)
    args.add_argument("--epochs", type=int, default=1000)
    args.add_argument("--iter-per-epoch", type=int, default=150)
    args.add_argument("--adv-steps", type=int, default=10)
    args.add_argument("--aux-steps", type=int, default=5)
    args.add_argument("--scheduling", type=bool, default=True)
    args.add_argument("--use-threshold", type=bool, default=True)
    args.add_argument("--anneal", type=float, default=0.5)
    args.add_argument("--train-seed", type=int, default=2)
    args.add_argument("--scaleX", type=float, default=2.0)
    args.add_argument("--scaleG", type=float, default=4.0)
    args.add_argument("--coefX", type=float, default=4.0)
    args.add_argument("--coefG", type=float, default=1.0)
    args.add_argument("--current", type=str, default=None)
    args = args.parse_args()

    # Set current directory if not provided
    if args.current is None:
        args.current = str(datetime.datetime.now()).replace(" ", "_")
    
    ### initialize simulation session
    simul = session(device=device,
                dist_family=dist_family,
                learning_rate=args.learning_rate,
                hidden_dim=args.hidden_dim,
                enc_num_layers=args.enc_layers,
                div_num_layers=args.div_layers,
                seed=args.seed,
                init_seed=args.init_seed,
                current=args.current,
                scaleX=args.scaleX,
                scaleG=args.scaleG,
                coefX=args.coefX,
                coefG=args.coefG)

    print("\nPENALTY NAME: " + args.penalty)
    simul.train(args)