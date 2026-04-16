#!/usr/bin/env python

import inox
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import torch
import os

from datasets import load_from_disk
from tqdm import trange

# isort: split
from utils import *


CONFIG = {
    'n_measurements': 128,
    'total_pixels': 784,
    'hid_channels': (128, 256, 384),
    'hid_blocks': (5, 5, 5),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {1: 4},
    'dropout': 0.1,
    'sampler': 'ddpm',
    'sde': {'a': 1e-3, 'b': 1e2},
    'heuristic': 'cov_x',
    'discrete': 256,
    'maxiter': 1,
    'epochs': 256,
    'batch_size': 256,
    'lr_init': 2e-4,
    'ema_decay': 0.9999,
}


def make_A_on_the_fly(idx, total_pixels, n_measurements):
    gen = torch.Generator(device='cpu').manual_seed(int(idx))
    A = torch.randn(total_pixels, n_measurements, generator=gen)
    A = A / torch.norm(A, dim=0, keepdim=True).clamp_min(1e-12)
    return A.numpy()


def generate_local(model, dataset, rng, batch_size, **kwargs):
    # Standard loop instead of dataset.map to avoid multiprocessing headaches
    indices = np.arange(len(dataset))
    y_data = np.array(dataset['y']).squeeze(1)

    # Recreate A for the batch
    A_batch = np.stack([make_A_on_the_fly(i, 784, 128) for i in indices[:batch_size]])
    y_batch = y_data[:batch_size]

    # This is where your sampling logic runs
    x = sample(model, y_batch, A_batch, rng.split(), **kwargs)
    return x.reshape(-1, 28, 28, 1)


def run_training():
    # Initialize WandB once
    runid = 'mnist_local_' + wandb.util.generate_id()
    run = wandb.init(project='mnist-flow-matching', id=runid, config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    # JAX Setup
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    rng = inox.random.PRNG(42)
    sde = VESDE(**CONFIG['sde'])

    # Load Data
    dataset = load_from_disk(PATH / f'hf/mnist-linear-128')
    dataset.set_format('numpy')

    # --- START LAPS ---
    model = None

    for lap in range(32):
        print(f"\n--- STARTING LAP {lap} ---")

        # 1. Initialization / Load Previous
        if lap > 0:
            model = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
        else:
            # Fit moments for lap 0
            y_fit = dataset['train'][:16384]['y'].squeeze(1)
            A_fit = np.stack([make_A_on_the_fly(i, 784, 128) for i in range(16384)])
            y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)

            mu_x, cov_x = fit_moments(
                features=784, rank=64, shard=True,
                A=inox.Partial(measure, A_fit), y=y_fit,
                cov_y=1e-3 ** 2, sampler='ddim', sde=sde,
                steps=256, key=rng.split()
            )
            model = GaussianDenoiser(mu_x, cov_x)

        # 2. Training Loop (Simplified)
        # ... [Insert your compiled JAX sgd_step logic here] ...

        # 3. Save Checkpoint
        dump_module(model, runpath / f'checkpoint_{lap}.pkl')
        print(f"Lap {lap} complete. Checkpoint saved.")


if __name__ == '__main__':
    run_training()