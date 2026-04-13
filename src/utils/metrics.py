"""
Metrics for evaluating SNN energy efficiency and accuracy.

Tracks spike rates, synaptic operations, estimated energy cost,
and classification accuracy across training and evaluation.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EnergyMetrics:
    """Immutable container for energy-related metrics from a single forward pass."""
    total_spikes: int
    total_possible_spikes: int
    spike_rate: float
    synaptic_operations: int
    estimated_energy_joules: float
    accuracy: float
    loss: float

    @property
    def energy_per_correct(self) -> float:
        """Estimated energy per correctly classified sample."""
        if self.accuracy == 0:
            return float("inf")
        return self.estimated_energy_joules / max(self.accuracy, 1e-8)


# Energy cost constants (approximate, based on neuromorphic literature)
# Loihi 2 estimates: ~23 pJ per synaptic operation, ~120 pJ per spike
ENERGY_PER_SPIKE_PJ = 120.0
ENERGY_PER_SYNOP_PJ = 23.0
PJ_TO_JOULES = 1e-12


def compute_spike_rate(spike_recordings: list[torch.Tensor]) -> float:
    """
    Compute average spike rate across all layers and timesteps.

    Args:
        spike_recordings: List of tensors, each shape [timesteps, batch, neurons]

    Returns:
        Average fraction of neurons firing per timestep (0.0 to 1.0)
    """
    total_spikes = 0
    total_possible = 0
    for spikes in spike_recordings:
        total_spikes += spikes.sum().item()
        total_possible += spikes.numel()
    if total_possible == 0:
        return 0.0
    return total_spikes / total_possible


def estimate_synaptic_operations(
    spike_recordings: list[torch.Tensor],
    fan_outs: list[int],
) -> int:
    """
    Estimate total synaptic operations (spike x fan-out connections).

    Each spike in layer i triggers fan_out[i] synaptic operations
    in the downstream layer.

    Args:
        spike_recordings: List of spike tensors per layer
        fan_outs: Number of outgoing connections per neuron in each layer

    Returns:
        Total estimated synaptic operations
    """
    total_ops = 0
    for spikes, fan_out in zip(spike_recordings, fan_outs):
        layer_spikes = int(spikes.sum().item())
        total_ops += layer_spikes * fan_out
    return total_ops


def estimate_energy(
    spike_recordings: list[torch.Tensor],
    fan_outs: list[int],
) -> float:
    """
    Estimate total energy consumption in joules for a forward pass.

    Uses approximate Loihi 2 energy figures:
    - 120 pJ per spike generation
    - 23 pJ per synaptic operation

    Args:
        spike_recordings: List of spike tensors per layer
        fan_outs: Number of outgoing connections per neuron in each layer

    Returns:
        Estimated energy in joules
    """
    total_spikes = sum(int(s.sum().item()) for s in spike_recordings)
    syn_ops = estimate_synaptic_operations(spike_recordings, fan_outs)

    energy_pj = (total_spikes * ENERGY_PER_SPIKE_PJ
                 + syn_ops * ENERGY_PER_SYNOP_PJ)
    return energy_pj * PJ_TO_JOULES


def compute_all_metrics(
    spike_recordings: list[torch.Tensor],
    fan_outs: list[int],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_value: float,
) -> EnergyMetrics:
    """
    Compute all metrics for a single batch/epoch.

    Args:
        spike_recordings: List of spike tensors per layer
        fan_outs: Fan-out per layer
        predictions: Model output predictions
        targets: Ground truth labels
        loss_value: Scalar loss value

    Returns:
        Immutable EnergyMetrics dataclass
    """
    total_spikes = sum(int(s.sum().item()) for s in spike_recordings)
    total_possible = sum(s.numel() for s in spike_recordings)
    spike_rate = total_spikes / max(total_possible, 1)
    syn_ops = estimate_synaptic_operations(spike_recordings, fan_outs)
    energy = estimate_energy(spike_recordings, fan_outs)

    predicted_labels = predictions.argmax(dim=1)
    correct = (predicted_labels == targets).sum().item()
    accuracy = correct / max(len(targets), 1)

    return EnergyMetrics(
        total_spikes=total_spikes,
        total_possible_spikes=total_possible,
        spike_rate=spike_rate,
        synaptic_operations=syn_ops,
        estimated_energy_joules=energy,
        accuracy=accuracy,
        loss=loss_value,
    )
