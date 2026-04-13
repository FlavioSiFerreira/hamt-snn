"""
Data loading utilities for neuromorphic datasets.

Supports N-MNIST, DVS-Gesture, SHD (Spiking Heidelberg Digits),
and CIFAR10-DVS via the tonic library with fallback to direct download.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import Optional
import os


# Default data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_static_mnist(
    batch_size: int = 128,
    num_steps: int = 25,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Load static MNIST encoded as rate-coded spike trains.

    This is the simplest benchmark: standard MNIST images converted
    to Poisson spike trains. Good for initial proof of concept because
    it requires no special neuromorphic dataset downloads.

    Args:
        batch_size: Batch size for DataLoader
        num_steps: Number of simulation timesteps
        data_dir: Override default data directory

    Returns:
        Tuple of (train_loader, test_loader)
    """
    from torchvision import datasets, transforms

    save_dir = data_dir or DATA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])

    train_ds = datasets.MNIST(
        root=str(save_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root=str(save_dir),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, test_loader


def rate_encode(
    images: torch.Tensor,
    num_steps: int = 25,
) -> torch.Tensor:
    """
    Convert pixel intensities to Poisson spike trains.

    Each pixel value is treated as a firing probability per timestep.

    Args:
        images: Tensor of shape [batch, channels, height, width], values in [0, 1]
        num_steps: Number of timesteps to generate

    Returns:
        Spike tensor of shape [num_steps, batch, channels * height * width]
        with binary values (0 or 1)
    """
    batch_size = images.shape[0]
    flat = images.view(batch_size, -1)

    # Clamp to valid probability range
    probs = flat.clamp(0.0, 1.0)

    # Generate Poisson spikes: random < probability = spike
    spikes = torch.rand(num_steps, batch_size, probs.shape[1], device=probs.device) < probs.unsqueeze(0)
    return spikes.float()
