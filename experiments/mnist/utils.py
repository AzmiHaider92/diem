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

def measure(A: Array, x: Array) -> Array:
    """Linear projection forward model: y = x @ A"""
    # x: (batch, 784), A: (batch, 784, 100) -> y: (batch, 100)
    return jnp.einsum('...i,...ij->...j', x, A)

def sample(
    model: nn.Module,
    y: Array,
    A: Array,
    key: Array,
    shard: bool = False,
    img_size: int = 28,
    **kwargs,
) -> Array:
    if shard:
        y, A = distribute((y, A))

    x = sample_any(
        model=model,
        shape=(img_size**2,),
        shard=shard,
        A=inox.Partial(measure, A),
        y=y.reshape(y.shape[0], -1), # Ensure y is flat
        cov_y=1e-3**2,
        key=key,
        **kwargs,
    )

    return unflatten(x, width=img_size, height=img_size)

def make_model(
    key: Array,
    hid_channels: Sequence[int] = (128, 256, 384),
    hid_blocks: Sequence[int] = (5, 5, 5),
    kernel_size: Sequence[int] = (3, 3),
    emb_features: int = 256,
    heads: Dict[int, int] = {1: 4},
    dropout: float = 0.1,
    img_size: int = 28,
    **absorb,
) -> Denoiser:
    return Denoiser(
        network=FlatUNet(
            in_channels=1, # Grayscale
            out_channels=1,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            emb_features=emb_features,
            heads=heads,
            dropout=dropout,
            img_size=img_size,
            key=key,
        ),
        emb_features=emb_features,
    )

class FlatUNet(UNet):
    img_size: int = 28
    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        x = unflatten(x, width=self.img_size, height=self.img_size)
        x = super().__call__(x, t, key)
        return flatten(x)