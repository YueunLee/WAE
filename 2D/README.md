# 2D Wasserstein Autoencoder (WAE) with Exact Penalty Methods

This repository implements a **2D** Wasserstein Autoencoder (WAE) to demonstrate that solving the WAE problem with exact penalties yields the true squared 2-Wasserstein distance.

## Overview

This project focuses on a 2-dimensional data distribution, employing truncated multivariate normal distributions. It utilizes Input Convex Neural Networks (ICNN) for transport maps and RealNVP for bijective transformations.

Let $P_X$ be the data distribution, $\psi$ be the Brenier potential, and $g$ be a bijective decoder.
The prior distribution is defined as the transformed data distribution, $(g^{-1} \circ \nabla \psi)_{\sharp} P_X$, implying that the optimal encoder corresponds to $g^{-1} \circ \nabla \psi$. This formulation leads to the following equality:
$$
    W_2^2(P_X, (\nabla \psi)_{\sharp}P_X) = W_2^2(P_X, (g \circ (g^{-1} \circ \nabla \psi))_{\sharp}P_X) = W_2^2(P_X, g_{\sharp}P_Z).
$$

The encoder is parameterized using a RealNVP architecture and applied as $q \circ \nabla \psi$, where $\psi$ is fixed. The primary learning objective is to approximate $g^{-1}$.

Due to the support constraints of $f$-divergences, we focus on the Jensen-Shannon divergence and the squared Hellinger distance. The equality constraint between the aggregated posterior and the prior distribution is verified using an equivalence test. This test is based on the [B-test](http://papers.neurips.cc/paper/5081-b-test-a-non-parametric-low-variance-kernel-two-sample-test.pdf), a non-parametric two-sample test utilizing the squared Maximum Mean Discrepancy (MMD).

Training terminates upon passing the equivalence test. Notably, when the square root of an $f$-divergence is used as a penalty, the reconstruction loss closely approximates the true Wasserstein distance.

## Installation

Ensure Python 3.8+ and PyTorch are installed. Install dependencies:

```bash
pip install torch numpy matplotlib seaborn
```

## Usage

### Single Run

Run the main script with defaults:

```bash
python main.py
```

Key arguments:
- `--penalty`: Penalty type (default: 'fgan_js')
- `--anneal`: Penalty coefficient annealing (default: 1.0)
- `--epochs`: Training epochs (default: 500)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--batch-size`: Batch size (default: 256)
- `--current`: Custom result directory

### Batch Runs

Execute multiple configurations:

```bash
./simul_fdiv.bash
./simul_sqrt_fdiv.bash
```

This runs penalty and anneal combinations, saving to a timestamped results directory.

## Files Description

- **`main.py`**: Main training script. Sets up models, runs training, evaluates with MMD/KS tests, and generates plots.
- **`architecture.py`**: Model architectures: DenseICNN (convex potential), RealNVP (bijective flow), MLPBlock (discriminator).
- **`penalties.py`**: Penalty functions for adversarial training, including f-divergences.
- **`utils.py`**: Utilities for distributions, Wasserstein distance, KS tests, and plotting.
- **`mmd.py`**: MMD batched equivalence test for distribution comparison.
- **`TruncNormal.py`**: Truncated multivariate normal distributions.
- **`simul_fdiv.bash`**: Bash script for batch experiments.
