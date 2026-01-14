# 1D Wasserstein Autoencoder (WAE) and Exact Penalty Method

This repository contains the implementation of a 1D Wasserstein Autoencoder (WAE) for generative modeling and distribution matching. The WAE uses optimal transport principles to learn a latent representation that aligns with a prior distribution.

## Overview

This project focuses on a 1D simulation using truncated normal distributions for both data and model distributions. The prior distribution is set to the uniform distribution on \[0,1\].
Various $f$-divergence penalties are employed as penalty (or regularization) terms.

In particular, the square roots of $f$-divergences act as exact penalties. This ensures that the minimum of the penalized problem equals the minimum of the original constrained problem, while the solution satisfies the constraint.

The equality constraint between the aggregate posterior and the prior distribution is verified using an equivalence test. If the upper bound of the confidence interval falls below a predefined threshold, the two distributions are considered statistically identical.

Training terminates when the equivalence test is passed. Notably, when the square root of an $f$-divergence is used as a penalty, the reconstruction loss closely approximates the true Wasserstein distance upon convergence.


## Installation

Ensure you have Python 3.8+ and PyTorch installed. Install required packages:

```bash
pip install torch numpy matplotlib
```

## Usage

### Single Run

Run the main training script with default settings:

```bash
python main.py
```

Available arguments:
- `--penalty`: Penalty function (default: 'fgan_js')
- `--anneal`: Annealing rate for penalty coefficient (default: 0.5)
- `--epochs`: Number of training epochs (default: 1000)
- `--learning-rate`: Learning rate (default: 0.0005)
- `--batch-size`: Batch size (default: 256)
- `--train-seed`: Random seed for training (default: 2)
- `--current`: Custom result directory path (default: timestamp)

### Batch Runs

Use the provided bash script to run multiple experiments:

```bash
./simul_fdiv.bash
./simul_sqrt_fdiv.bash
```

This script runs combinations of penalties and anneal options, saving results to a timestamped directory.

## Files Description

- **`main.py`**: Main training script. Initializes the session, sets up models, and runs training with evaluation.
- **`penalties.py`**: Defines various penalty functions for adversarial training, including f-divergence penalties.
- **`structure.py`**: Contains the `session` class, which manages the WAE model components (encoder, decoder, discriminator, etc.).
- **`TruncNormal.py`**: Implements truncated normal distributions for data and prior sampling.
- **`utils.py`**: Utility functions for initialization, distance calculations, and plotting.
- **`simul_fdiv.bash`**: Bash script for batch execution of multiple penalty configurations.
