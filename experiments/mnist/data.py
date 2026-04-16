#!/usr/bin/env python

import os
import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset, Features, Array2D
from pathlib import Path

# Configuration for corruption
N_MEASUREMENTS = 100
TOTAL_PIXELS = 784  # 28 * 28 for MNIST
PATH = Path('.')


def get_projection_matrix(key, total_pixels=TOTAL_PIXELS, n_measurements=N_MEASUREMENTS):
    """
    Generates the A matrix for a specific image.
    Matches the logic of: A / ||A||
    """
    A = jax.random.normal(key, (total_pixels, n_measurements))
    # Using positional arguments for clip (val, min, max) to avoid keyword issues
    norm = jnp.linalg.norm(A, axis=0, keepdims=True)
    A = A / jnp.clip(norm, 1e-12, None)
    return A


def compress_mnist():
    print(f"--- Preparing MNIST with {N_MEASUREMENTS} linear measurements ---")

    # 1. Load raw MNIST
    dataset = load_dataset('mnist')

    # 2. Define the transformation
    def transform(batch, indices):
        images = np.array([np.array(img).flatten() for img in batch['image']], dtype=np.float32) / 255.0
        x = jnp.array(images)

        def apply_corruption(xi, idx):
            key = jax.random.PRNGKey(idx)
            A = get_projection_matrix(key)
            # Projection y = x^T A results in (100,)
            yi = xi @ A
            # Reshape to the "image" format (10, 10)
            return yi.reshape(10, 10)

        y_batch = jax.vmap(apply_corruption)(x, jnp.array(indices))

        return {
            'y': np.array(y_batch),
            'idx': np.array(indices)[:, None, None]
        }

    # 3. Apply transformation
    # Change the shape to be 2D (1 row, 100 columns)
    types = Features({
        'y': Array2D(shape=(10, 10), dtype='float32'),
        'idx': Array2D(shape=(1, 1), dtype='int32'),  # Keeping it 2D to be safe
    })

    processed_dataset = dataset.map(
        transform,
        with_indices=True,
        batched=True,
        batch_size=1000,
        remove_columns=['image', 'label'],
        features=types
    )

    # 4. Save to disk
    output_path = PATH / f'hf/mnist-linear-{N_MEASUREMENTS}'
    processed_dataset.save_to_disk(output_path)
    print(f"--- Dataset saved to {output_path} ---")


if __name__ == '__main__':
    os.makedirs(PATH, exist_ok=True)
    compress_mnist()