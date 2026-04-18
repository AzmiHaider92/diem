#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import os

from datasets import Array3D, Features, load_from_disk
from functools import partial
from tqdm import trange, tqdm
from typing import *
from pathlib import Path

# isort: split
from utils import *


CONFIG = {
    # Data
    'corruption': 75,
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


def generate(model, dataset, rng, batch_size, **kwargs):
    def transform(batch):
        y, A = batch['y'], batch['A']
        # Sample using the provided model
        x = sample(model, y, A, rng.split(), **kwargs)
        return {'x': np.asarray(x)}

    types = {'x': Array3D(shape=(32, 32, 3), dtype='float32')}

    return dataset.map(
        transform,
        features=Features(types),
        remove_columns=['y', 'A'],
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
    )


def train_lap(run, runpath, lap: int, rng: inox.random.PRNG):
    config = run.config

    # SDE Setup
    sde = VESDE(**CONFIG.get('sde'))

    # Data Loading
    dataset = load_from_disk(f'hf/cifar-mask-{config.corruption}')
    dataset.set_format('numpy')

    trainset_yA = dataset['train']
    testset_yA = dataset['test']

    # Evaluation subset
    y_eval, A_eval = testset_yA[:16]['y'], testset_yA[:16]['A']

    # Initialization/Previous Model Loading
    if lap > 0:
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
    else:
        # Fit initial moments for the first lap
        y_fit, A_fit = trainset_yA[:16384]['y'], trainset_yA[:16384]['A']

        mu_x, cov_x = fit_moments(
            features=32 * 32 * 3,
            rank=64,
            shard=False,
            A=inox.Partial(measure, A_fit),
            y=flatten(y_fit),
            cov_y=1e-3 ** 2,
            sampler='ddim',
            sde=sde,
            steps=256,
            maxiter=None,
            key=rng.split(),
        )
        del y_fit, A_fit
        previous = GaussianDenoiser(mu_x, cov_x)

    ## 1. Generate Synthetic Dataset from Previous Model
    print(f"\n--- Lap {lap}: Generating data ---")
    trainset = generate(
        model=previous,
        dataset=trainset_yA,
        rng=rng,
        batch_size=config.batch_size,
        shard=False,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )
    testset = generate(
        model=previous,
        dataset=testset_yA,
        rng=rng,
        batch_size=config.batch_size,
        shard=False,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )

    ## 2. Fit Moments
    x_fit = flatten(trainset[:16384]['x'])
    mu_x, cov_x = ppca(x_fit, rank=64, key=rng.split())
    del x_fit

    # Initialize New Model
    if lap > 0:
        model = previous
    else:
        model = make_model(key=rng.split(), **CONFIG)

    model.mu_x = mu_x

    # Apply Heuristics
    if config.heuristic == 'zeros':
        model.cov_x = jnp.zeros_like(mu_x)
    elif config.heuristic == 'ones':
        model.cov_x = jnp.ones_like(mu_x)
    elif config.heuristic == 'cov_t':
        model.cov_x = jnp.ones_like(mu_x) * 1e6
    elif config.heuristic == 'cov_x':
        model.cov_x = cov_x

    model.train(True)
    static, params, others = model.partition(nn.Parameter)

    # Optimizer & EMA
    objective = DenoiserLoss(sde=sde)
    total_steps = config.epochs * len(trainset_yA) // config.batch_size
    optimizer = Adam(steps=total_steps, **config)
    opt_state = optimizer.init(params)
    ema = EMA(decay=config.ema_decay)
    avrg = params

    # Training Utilities
    @jax.jit
    @jax.vmap
    def augment(x, key):
        keys = jax.random.split(key, 3)
        x = random_flip(x, keys[0], axis=-2)
        x = random_hue(x, keys[1], delta=1e-2)
        x = random_saturation(x, keys[2], lower=0.95, upper=1.05)
        return x

    @jax.jit
    def ell(params, others, x, key):
        keys = jax.random.split(key, 3)
        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])
        return objective(static(params, others), x, z, t, key=keys[2])

    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)
        return loss, avrg, params, opt_state

    ## 3. Training Loop
    print(f"--- Lap {lap}: Training ---")
    for epoch in (bar := trange(config.epochs, ncols=88)):
        loader = trainset.shuffle(seed=int(rng.split()[0])).iter(
            batch_size=config.batch_size, drop_last_batch=True
        )

        train_losses = []
        for batch in prefetch(loader):
            x = augment(batch['x'], rng.split(len(batch['x'])))
            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, flatten(x), key=rng.split())
            train_losses.append(loss)

        loss_train = np.mean(train_losses)

        # Validation
        val_loader = testset.iter(batch_size=config.batch_size, drop_last_batch=True)
        val_losses = [ell(avrg, others, flatten(b['x']), key=rng.split()) for b in prefetch(val_loader)]
        loss_val = np.mean(val_losses)

        bar.set_postfix(loss=loss_train, loss_val=loss_val)

        # Periodic Logging & Sampling
        if (epoch + 1) % 16 == 0:
            model_eval = static(avrg, others)
            model_eval.train(False)
            x_samples = sample(
                model=model_eval, y=y_eval, A=A_eval, key=rng.split(),
                shard=False, sampler=config.sampler, steps=config.discrete, maxiter=config.maxiter
            )
            run.log({
                'lap': lap,
                'loss': loss_train,
                'loss_val': loss_val,
                'samples': wandb.Image(to_pil(x_samples.reshape(4, 4, 32, 32, 3), zoom=4)),
            })
        else:
            run.log({'lap': lap, 'loss': loss_train, 'loss_val': loss_val})

    # Save Checkpoint
    final_model = static(avrg, others)
    final_model.train(False)
    dump_module(final_model, Path(runpath + f'/checkpoint_{lap}.pkl'))


if __name__ == '__main__':
    # Initialize WandB
    runid = 'cifar_diem_' + wandb.util.generate_id()

    run = wandb.init(
        project='mnist-flow-matching',
        id=runid,
        config=CONFIG,
    )

    runpath = f'runs/{run.name}_{run.id}'
    os.makedirs(runpath, exist_ok=True)

    # Local RNG
    rng = inox.random.PRNG(42)

    # Run the iterative laps sequentially
    for lap in range(32):
        train_lap(run, runpath, lap, rng)

    run.finish()