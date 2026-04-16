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
total_pixels = 28*28

CONFIG = {
    # Data
    'n_measurements': 100,
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
    'heuristic': None,
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
    batch_size = kwargs.get('batch_size', 64)

    def transform(batch, indices):
        y = np.array(batch['y']).reshape(len(indices), -1)
        idx_seeds = np.array(batch['idx']).flatten()

        A = np.stack([
            make_A_on_the_fly(s, CONFIG['total_pixels'], CONFIG['n_measurements'])
            for s in idx_seeds
        ])

        x = sample(model, y, A, rng.split(), **kwargs)
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
    dataset = load_from_disk(PATH / f'hf/mnist-linear-{config.n_measurements}')
    dataset.set_format('numpy')

    # Prep Eval Data
    eval_batch = dataset['test'][:16]
    y_eval = eval_batch['y'].reshape(16, -1)
    idx_eval = eval_batch['idx'].flatten()
    A_eval = np.stack([make_A_on_the_fly(i, 784, config.n_measurements) for i in idx_eval])
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

    if lap > 0:
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
    else:
        fit_batch = dataset['train'][:16384]
        y_fit = fit_batch['y'].reshape(16384, -1)
        idx_fit = fit_batch['idx'].flatten()
        A_fit = np.stack([make_A_on_the_fly(i, 784, config.n_measurements) for i in idx_fit])
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)

        mu_x, cov_x = fit_moments(
            features=784, rank=64, shard=True,
            A=inox.Partial(measure, A_fit), y=y_fit,
            cov_y=1e-3 ** 2, sampler='ddim', sde=sde,
            steps=256, maxiter=None, key=rng.split(),
        )
        previous = GaussianDenoiser(mu_x, cov_x)

    # Distributed Setup
    static, arrays = previous.partition()
    previous = static(jax.device_put(arrays, replicated))

    print(f"\n--- Lap {lap}: Generating Synthetic Data ---")
    trainset = generate(previous, dataset['train'], rng, shard=True, **config)
    testset = generate(previous, dataset['test'], rng, shard=True, **config)

    # Model Initialization
    x_fit = flatten(trainset[:16384]['x'])
    mu_x, cov_x = ppca(x_fit, rank=64, key=rng.split())

    model = previous if lap > 0 else make_model(key=rng.split(), **config)
    model.mu_x, model.cov_x = mu_x, cov_x
    model.train(True)

    static, params, others = model.partition(nn.Parameter)
    objective = DenoiserLoss(sde=sde)
    optimizer = Adam(steps=config.epochs * len(dataset['train']) // config.batch_size, **config)
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
            x_samp = sample(model_eval, y_eval, A_eval, rng.split(), shard=True, **config)
            run.log({'lap': lap, 'loss': np.mean(losses),
                     'samples': wandb.Image(to_pil(x_samp.reshape(4, 4, 28, 28, 1), zoom=4))})
        else:
            run.log({'lap': lap, 'loss': np.mean(losses)})

    dump_module(static(avrg, others), runpath / f'checkpoint_{lap}.pkl')


if __name__ == '__main__':
    runid = 'mnist_diem_' + wandb.util.generate_id()

    run = wandb.init(
        project='mnist-flow-matching',
        id=runid,
        config=CONFIG,
    )

    runpath = f'runs/{run.name}_{run.id}'
    os.makedirs(runpath, exist_ok=True)

    rng = inox.random.PRNG(42)

    for lap in range(32):
        train_lap(run, runpath, lap, rng)