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

def sample(
    model: nn.Module,
    y: Array,
    A: Array,
    key: Array,
    shard: bool = False,
    sampler: str = 'ddpm',
    sde: Any = None,
    steps: int = 256,
    maxiter: int = 1,
) -> Array:
    if shard:
        y, A = distribute((y, A))

    x = sample_any(
        model=model,
        shape=(784,),
        shard=shard,
        A=inox.Partial(measure, A),
        y=y,
        cov_y=1e-3**2,
        key=key,
        sampler=sampler,
        sde=sde,
        steps=steps,
        maxiter=maxiter,
    )

    return unflatten(x, width=28, height=28)

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