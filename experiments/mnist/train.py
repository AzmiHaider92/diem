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
total_pixels = 784
n_measurements = 100

CONFIG = {
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


def make_A_on_the_fly(idx):
    gen = torch.Generator(device='cpu').manual_seed(int(idx))
    A = torch.randn(total_pixels, n_measurements, generator=gen)
    A = A / torch.norm(A, dim=0, keepdim=True).clamp_min(1e-12)
    return A.numpy()


def generate(model, dataset, rng, batch_size, shard, sampler, sde, steps, maxiter):
    def transform(batch, indices):
        y = np.array(batch['y']).reshape(len(indices), -1)
        idx_seeds = np.array(batch['idx']).flatten()

        A = np.stack([make_A_on_the_fly(s) for s in idx_seeds])

        x = sample(
            model=model, y=y, A=A, key=rng.split(),
            shard=shard, sampler=sampler, sde=sde, steps=steps, maxiter=maxiter
        )
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
    sde = VESDE(**CONFIG.get('sde'))

    dataset = load_from_disk(PATH / f'hf/mnist-linear-{n_measurements}')
    dataset.set_format('numpy')
    trainset_yA = dataset['train']
    testset_yA = dataset['test']

    # Eval data
    eval_data = testset_yA[:16]

    # y_eval should be (16, 10, 10) from the dataset, reshape to (16, 100)
    y_eval = eval_data['y'].reshape(16, 100)

    # idx_eval should be (16,)
    idx_eval = eval_data['idx'].flatten()

    # A_eval should be (16, 784, 100)
    A_eval = np.stack([make_A_on_the_fly(i) for i in idx_eval])

    if lap > 0:
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
    else:
        fit_data = trainset_yA[:16384]
        y_fit = fit_data['y'].reshape(16384, -1)
        A_fit = np.stack([make_A_on_the_fly(i) for i in fit_data['idx'].flatten()])

        mu_x, cov_x = fit_moments(
            features=total_pixels, rank=64, shard=False,
            A=inox.Partial(measure, A_fit), y=y_fit,
            cov_y=1e-3 ** 2, sampler='ddim', sde=sde,
            steps=256, maxiter=None, key=rng.split(),
        )
        previous = GaussianDenoiser(mu_x, cov_x)

    print(f"\n--- Lap {lap}: Generating data ---")
    gen_args = {
        'model': previous, 'dataset': trainset_yA, 'rng': rng,
        'batch_size': config.batch_size, 'shard': False, 'sampler': config.sampler,
        'sde': sde, 'steps': config.discrete, 'maxiter': config.maxiter
    }
    trainset = generate(**gen_args)
    testset = generate(**{**gen_args, 'dataset': testset_yA})

    x_fit = flatten(trainset[:16384]['x'])
    mu_x, cov_x = ppca(x_fit, rank=64, key=rng.split())

    model = previous if lap > 0 else make_model(key=rng.split(), **CONFIG)
    model.mu_x, model.cov_x = mu_x, cov_x
    model.train(True)

    static, params, others = model.partition(nn.Parameter)
    objective = DenoiserLoss(sde=sde)
    optimizer = Adam(steps=config.epochs * len(trainset_yA) // config.batch_size, **config)
    opt_state = optimizer.init(params)
    ema = EMA(decay=config.ema_decay)
    avrg = params

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
        losses = [sgd_step(avrg, params, others, opt_state, flatten(batch['x']), rng.split())[0] for batch in
                  prefetch(loader)]

        if (epoch + 1) % 16 == 0:
            model_eval = static(avrg, others)
            model_eval.train(False)
            # After (Explicitly check that y_eval and A_eval are batches of 16)
            x_samples = sample(
                model=model_eval,
                y=y_eval,  # Should be (16, 100)
                A=A_eval,  # Should be (16, 784, 100)
                key=rng.split(),
                shard=False,
                sampler=config.sampler,
                sde=sde,
                steps=config.discrete,
                maxiter=config.maxiter
            )

            run.log({'lap': lap, 'loss': np.mean(losses),
                     'samples': wandb.Image(to_pil(x_samples.reshape(4, 4, 28, 28, 1), zoom=4))})
        else:
            run.log({'lap': lap, 'loss': np.mean(losses)})

    dump_module(static(avrg, others), runpath / f'checkpoint_{lap}.pkl')


if __name__ == '__main__':
    run = wandb.init(project='mnist-flow-matching', id='mnist_diem_' + wandb.util.generate_id(), config=CONFIG)
    runpath = Path(f"runs/{run.name}_{run.id}")
    runpath.mkdir(parents=True, exist_ok=True)
    rng = inox.random.PRNG(42)
    for lap in range(32):
        train_lap(run, runpath, lap, rng)