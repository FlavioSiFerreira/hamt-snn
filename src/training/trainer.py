"""
Training loops for baseline SNN and HAMT-SNN.

Handles the full training pipeline: data loading, forward pass,
loss computation, backprop, and metric logging. Supports both
standard training (baseline) and metabolic-aware training (HAMT).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from typing import Optional

from ..utils.metrics import compute_all_metrics, estimate_energy, EnergyMetrics
from ..utils.data import rate_encode
from ..losses.metabolic_loss import MetabolicLoss


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 30,
    lr: float = 5e-4,
    num_steps: int = 25,
    device: torch.device = torch.device("cpu"),
    save_dir: Optional[Path] = None,
) -> dict:
    """
    Train baseline SNN with standard cross-entropy loss.

    No energy awareness, no habituation. This is the control condition.

    Args:
        model: BaselineSNN instance
        train_loader: Training data
        test_loader: Test data
        num_epochs: Training epochs
        lr: Learning rate
        num_steps: SNN simulation timesteps
        device: CPU or CUDA device
        save_dir: Directory to save results

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "spike_rate": [],
        "energy_joules": [],
        "epoch_time_sec": [],
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_spike_rates = []
        epoch_energy = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Rate-encode images to spike trains
            spikes_in = rate_encode(images, num_steps).to(device)

            # Forward pass
            mem_out, spk_sum, spike_recs = model(spikes_in)

            # Loss on summed output spikes (rate coding readout)
            loss = loss_fn(spk_sum, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            predicted = spk_sum.argmax(dim=1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)
            epoch_loss += loss.item()

            # Energy metrics
            fan_outs = model.get_fan_outs()
            with torch.no_grad():
                sr = sum(s.sum().item() for s in spike_recs) / sum(
                    s.numel() for s in spike_recs
                )
                energy = estimate_energy(spike_recs, fan_outs)
                epoch_spike_rates.append(sr)
                epoch_energy.append(energy)

            if batch_idx % 100 == 0:
                print(
                    f"  Baseline Epoch {epoch+1}/{num_epochs} "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f} "
                    f"Spike Rate: {sr:.4f}",
                    flush=True,
                )

        # Epoch stats
        train_acc = epoch_correct / max(epoch_total, 1)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        avg_sr = sum(epoch_spike_rates) / max(len(epoch_spike_rates), 1)
        avg_energy = sum(epoch_energy) / max(len(epoch_energy), 1)

        # Test evaluation
        test_acc = evaluate(model, test_loader, num_steps, device)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["spike_rate"].append(avg_sr)
        history["energy_joules"].append(avg_energy)
        history["epoch_time_sec"].append(epoch_time)

        print(
            f"Baseline Epoch {epoch+1}/{num_epochs}: "
            f"Train Acc={train_acc:.4f} Test Acc={test_acc:.4f} "
            f"Spike Rate={avg_sr:.4f} Energy={avg_energy:.2e}J "
            f"Time={epoch_time:.1f}s",
            flush=True,
        )

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "baseline_history.json", "w") as f:
            json.dump(history, f, indent=2)
        torch.save(model.state_dict(), save_dir / "baseline_model.pt")

    return history


def train_hamt(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 30,
    lr: float = 5e-4,
    num_steps: int = 25,
    lambda_energy: float = 0.001,
    lambda_habituation: float = 0.0005,
    target_spike_rate: float = 0.05,
    ramp_epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    save_dir: Optional[Path] = None,
) -> dict:
    """
    Train HAMT-SNN with metabolic cost-aware loss.

    Uses the full HAMT loss: task + energy + habituation reward.

    Args:
        model: HAMTSNN instance
        train_loader: Training data
        test_loader: Test data
        num_epochs: Training epochs
        lr: Learning rate
        num_steps: SNN simulation timesteps
        lambda_energy: Energy penalty weight
        lambda_habituation: Habituation reward weight
        target_spike_rate: Target spike rate (0.05 = 5%, cortical level)
        ramp_epochs: Epochs to ramp up energy penalty
        device: CPU or CUDA device
        save_dir: Directory to save results

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MetabolicLoss(
        lambda_energy=lambda_energy,
        lambda_habituation=lambda_habituation,
        target_spike_rate=target_spike_rate,
        ramp_epochs=ramp_epochs,
    )

    history = {
        "train_loss": [],
        "train_loss_task": [],
        "train_loss_energy": [],
        "train_loss_habituation": [],
        "train_acc": [],
        "test_acc": [],
        "spike_rate": [],
        "energy_joules": [],
        "epoch_time_sec": [],
        "habituation_stats": [],
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()

        epoch_losses = {"task": 0, "energy": 0, "habituation": 0, "total": 0}
        epoch_correct = 0
        epoch_total = 0
        epoch_spike_rates = []
        epoch_energy = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            spikes_in = rate_encode(images, num_steps).to(device)

            # Forward pass (HAMT returns habituation strengths)
            mem_out, spk_sum, spike_recs, hab_strengths = model(spikes_in)

            # HAMT loss: task + energy + habituation
            loss, components = loss_fn(
                predictions=spk_sum,
                targets=labels,
                spike_recordings=spike_recs,
                habituation_strengths=hab_strengths,
                current_epoch=epoch,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            predicted = spk_sum.argmax(dim=1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)

            for key in epoch_losses:
                epoch_losses[key] += components.get(key, 0)

            fan_outs = model.get_fan_outs()
            with torch.no_grad():
                sr = sum(s.sum().item() for s in spike_recs) / sum(
                    s.numel() for s in spike_recs
                )
                energy = estimate_energy(spike_recs, fan_outs)
                epoch_spike_rates.append(sr)
                epoch_energy.append(energy)

            if batch_idx % 100 == 0:
                print(
                    f"  HAMT Epoch {epoch+1}/{num_epochs} "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss: {components['total']:.4f} "
                    f"(task={components['task']:.4f} "
                    f"energy={components['energy']:.4f} "
                    f"hab={components['habituation']:.4f}) "
                    f"SR: {sr:.4f}",
                    flush=True,
                )

        # Epoch stats
        n_batches = max(len(train_loader), 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        avg_sr = sum(epoch_spike_rates) / max(len(epoch_spike_rates), 1)
        avg_energy = sum(epoch_energy) / max(len(epoch_energy), 1)

        test_acc = evaluate_hamt(model, test_loader, num_steps, device)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(epoch_losses["total"] / n_batches)
        history["train_loss_task"].append(epoch_losses["task"] / n_batches)
        history["train_loss_energy"].append(epoch_losses["energy"] / n_batches)
        history["train_loss_habituation"].append(
            epoch_losses["habituation"] / n_batches
        )
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["spike_rate"].append(avg_sr)
        history["energy_joules"].append(avg_energy)
        history["epoch_time_sec"].append(epoch_time)
        history["habituation_stats"].append(model.get_habituation_stats())

        print(
            f"HAMT Epoch {epoch+1}/{num_epochs}: "
            f"Train Acc={train_acc:.4f} Test Acc={test_acc:.4f} "
            f"Spike Rate={avg_sr:.4f} Energy={avg_energy:.2e}J "
            f"Time={epoch_time:.1f}s",
            flush=True,
        )

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "hamt_history.json", "w") as f:
            json.dump(history, f, indent=2)
        torch.save(model.state_dict(), save_dir / "hamt_model.pt")

    return history


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    num_steps: int = 25,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluate baseline SNN accuracy on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            spikes_in = rate_encode(images, num_steps).to(device)
            _, spk_sum, _ = model(spikes_in)
            predicted = spk_sum.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / max(total, 1)


def evaluate_hamt(
    model: nn.Module,
    test_loader: DataLoader,
    num_steps: int = 25,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluate HAMT-SNN accuracy on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            spikes_in = rate_encode(images, num_steps).to(device)
            _, spk_sum, _, _ = model(spikes_in)
            predicted = spk_sum.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / max(total, 1)
