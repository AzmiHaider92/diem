r"""MNIST Linear Projection experiment helpers"""

import os
import jax.numpy as jnp
from jax import Array
from pathlib import Path
from typing import *

# isort: split
from diem.common import *
from diem.data import *
from diem.diffusion import *
from diem.image import *
from diem.nn import *
from diem.optim import *

PATH = Path(".")
PATH.mkdir(parents=True, exist_ok=True)

def measure(A: Array, x: Array) -> Array:
    """Linear projection forward model: y = x @ A"""
    return jnp.einsum('...i,...ij->...j', x, A)


def sample(model, y, A, key, shard=False, sampler='ddpm', sde=None, steps=256, maxiter=1):
    if shard:
        y, A = distribute((y, A))

    # y comes in as (16, 10, 10). We need it flat for the math: (16, 100)
    y_flat = y.reshape(y.shape[0], -1)

    x = sample_any(
        model=model,
        shape=(784,),        # This is the target: 28x28 = 784
        shard=shard,
        A=inox.Partial(measure, A),
        y=y_flat,            # This is the input: 10x10 = 100
        cov_y=1e-3**2,
        key=key,
        sampler=sampler,
        sde=sde,
        steps=steps,
        maxiter=maxiter,
    )

    # Return the batch of reconstructed images
    return x.reshape(-1, 28, 28, 1)

def make_model(key: Array, **config) -> Denoiser:
    return Denoiser(
        network=FlatUNet(
            in_channels=1,
            out_channels=1,
            hid_channels=config.get('hid_channels'),
            hid_blocks=config.get('hid_blocks'),
            kernel_size=config.get('kernel_size'),
            emb_features=config.get('emb_features'),
            heads=config.get('heads'),
            dropout=config.get('dropout'),
            key=key,
        ),
        emb_features=config.get('emb_features'),
    )

class FlatUNet(UNet):
    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        x = unflatten(x, width=28, height=28)
        x = super().__call__(x, t, key)
        return flatten(x)