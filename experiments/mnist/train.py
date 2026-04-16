#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import torch
import os

from datasets import Array3D, Features, load_from_disk
from functools import partial
from tqdm import trange
from typing import *
from pathlib import Path

# isort: split
from utils import *

PATH = Path(".")
total_pixels = 784  # 28*28
n_measurements = 100

CONFIG = {
    # Architecture
    'hid_channels': (128, 256, 384),
    'hid_blocks': (5, 5, 5),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {1: 4},
    'dropout': 0.1,
    # Sampling
    'sampler': 'ddpm',
    'sde': {'a': 1e-3, 'b': 1e2},
    'heuristic': 'cov_x',
    'discrete': 256,
    'maxiter': 1,
    # Training
    'epochs': 256,
    'batch_size': 64,
    'scheduler': 'constant',
    'lr_init': 2e-4,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
    'ema_decay': 0.9999,
}


def make_A_on_the_fly(idx, total_pixels, n_measurements):
    """Recreates the A matrix using the seed stored in the dataset."""
    gen = torch.Generator(device='cpu').manual_seed(int(idx))
    A = torch.randn(total_pixels, n_measurements, generator=gen)
    A = A / torch.norm(A, dim=0, keepdim=True).clamp_min(1e-12)
    return A.numpy()


def generate(model, dataset, rng, **kwargs):
    # Pull batch_size from kwargs to avoid double-passing later
    batch_size = kwargs.get('batch_size', 64)

    def transform(batch, indices):
        # Reshape y from (batch, 10, 10) to (batch, 100)
        y = np.array(batch['y']).reshape(len(indices), -1)
        idx_seeds = np.array(batch['idx']).flatten()

        A = np.stack([
            make_A_on_the_fly(s, total_pixels, n_measurements)
            for s in idx_seeds
        ])

        # Ensure sample doesn't get duplicate img_size or batch_size
        x = sample(model, y, A, rng.split(), img_size=28, **kwargs)
        return {'x': np.asarray(x).reshape(-1, 28, 28, 1)}

    types = {'x': Array3D(shape=(28, 28, 1), dtype='float32')}
    return dataset.map(
        transform,
        with_indices=True,
        features=Features(types),
        remove_columns=['y', 'idx'],
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
    )


def train_lap(run, runpath, lap: int, rng: inox.random.PRNG):
    config = run.config
    mesh = jax.sharding.Mesh(jax.local_devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    sde = VESDE(**CONFIG.get('sde'))
    dataset = load_from_disk(PATH / f'hf/mnist-linear-{n_measurements}')
    dataset.set_format('numpy')

    trainset_yA = dataset['train']
    testset_yA = dataset['test']

    # Evaluation subset preparation
    eval_indices = testset_yA[:16]['idx'].flatten()
    y_eval = testset_yA[:16]['y'].reshape(16, -1)

    def get_A_jax(idx):
        # Recreate A matrix using JAX for the eval set to keep it in the graph if needed
        # Note: Must match the logic of make_A_on_the_fly exactly
        key = jax.random.PRNGKey(idx.astype(jnp.int32))
        A = jax.random.normal(key, (total_pixels, n_measurements))
        A = A / jnp.clip(jnp.linalg.norm(A, axis=0, keepdims=True), 1e-12, None)
        return A

    A_eval = jax.vmap(get_A_jax)(jnp.array(eval_indices))
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

    # Initialization/Previous Model Loading
    if lap > 0:
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
    else:
        # Fit initial moments for Lap 0
        fit_data = trainset_yA[:16384]
        y_fit = fit_data['y'].reshape(16384, -1)
        idx_fit = fit_data['idx'].flatten()

        # We need A_fit for the GaussianDenoiser initial moments
        A_fit = jax.vmap(get_A_jax)(jnp.array(idx_fit))
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)

        mu_x, cov_x = fit_moments(
            features=total_pixels,
            rank=64,
            shard=True,
            A=inox.Partial(measure, A_fit),
            y=y_fit,
            cov_y=1e-3 ** 2,
            sampler='ddim',
            sde=sde,
            steps=256,
            maxiter=None,
            key=rng.split(),
        )
        previous = GaussianDenoiser(mu_x, cov_x)

    # Ensure previous model is sharded correctly
    static_prev, arrays_prev = previous.partition()
    previous = static_prev(jax.device_put(arrays_prev, replicated))

    print(f"\n--- Lap {lap}: Generating Synthetic Data ---")
    # Call generate without passing batch_size twice (once in config, once explicitly)
    trainset = generate(model=previous, dataset=trainset_yA, rng=rng, shard=False, **config)
    testset = generate(model=previous, dataset=testset_yA, rng=rng, shard=False, **config)

    # Moments & Model Initialization
    x_fit = flatten(trainset[:16384]['x'])
    mu_x, cov_x = ppca(x_fit, rank=64, key=rng.split())

    model = previous if lap > 0 else make_model(key=rng.split(), img_size=28, **config)
    model.mu_x, model.cov_x = mu_x, cov_x
    model.train(True)

    static, params, others = model.partition(nn.Parameter)
    objective = DenoiserLoss(sde=sde)
    optimizer = Adam(steps=config.epochs * len(trainset_yA) // config.batch_size, **config)
    opt_state = optimizer.init(params)
    ema = EMA(decay=config.ema_decay)
    avrg = params
    avrg, params, others, opt_state = jax.device_put((avrg, params, others, opt_state), replicated)

    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, key):
        def loss_fn(p):
            keys = jax.random.split(key, 3)
            z = jax.random.normal(keys[0], shape=x.shape)
            t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])
            return objective(static(p, others), x, z, t, key=keys[2])

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)
        return loss, avrg, params, opt_state

    print(f"--- Lap {lap}: Training ---")
    for epoch in (bar := trange(config.epochs, ncols=88)):
        loader = trainset.shuffle(seed=int(rng.split()[0])).iter(batch_size=config.batch_size, drop_last_batch=True)
        losses = []
        for batch in prefetch(loader):
            x = jax.device_put(flatten(batch['x']), distributed)
            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, rng.split())
            losses.append(loss)

        if (epoch + 1) % 16 == 0:
            model_eval = static(avrg, others)
            model_eval.train(False)
            # Sample with cleaned kwargs to avoid TypeError in DDPM
            x_samp = sample(model_eval, y_eval, A_eval, rng.split(), shard=True, img_size=28, **config)
            run.log({'lap': lap, 'loss': np.mean(losses),
                     'samples': wandb.Image(to_pil(x_samp.reshape(4, 4, 28, 28, 1), zoom=4))})
        else:
            run.log({'lap': lap, 'loss': np.mean(losses)})

    dump_module(static(avrg, others), runpath / f'checkpoint_{lap}.pkl')


if __name__ == '__main__':
    run = wandb.init(project='mnist-flow-matching', config=CONFIG)
    runpath = Path(f"runs/{run.name}_{run.id}")
    runpath.mkdir(parents=True, exist_ok=True)
    rng = inox.random.PRNG(42)

    for lap in range(32):
        train_lap(run, runpath, lap, rng)