"""
Advanced efficiency metrics that capture HAMT's advantage.

When baseline spike rates are already low (as with large hidden layers),
raw spike reduction may be modest. The real advantage of habituation is
SELECTIVITY: spikes are allocated to informative (novel) inputs rather
than wasted on predictable ones. These metrics capture that.
"""

import torch
import numpy as np
from typing import Optional


def accuracy_per_spike(
    accuracy: float,
    spike_rate: float,
) -> float:
    """
    Accuracy achieved per unit of spiking activity.

    Higher is better: same accuracy with fewer spikes, or better
    accuracy with same spikes.
    """
    if spike_rate < 1e-10:
        return float("inf") if accuracy > 0 else 0.0
    return accuracy / spike_rate


def accuracy_per_joule(
    accuracy: float,
    energy_joules: float,
) -> float:
    """
    Accuracy achieved per joule of estimated energy.

    The primary efficiency metric. HAMT should achieve higher
    accuracy-per-joule than baseline.
    """
    if energy_joules < 1e-20:
        return float("inf") if accuracy > 0 else 0.0
    return accuracy / energy_joules


def compute_temporal_efficiency(
    spike_recordings: list[torch.Tensor],
) -> dict[str, float]:
    """
    Measure how spike activity changes across timesteps within a sample.

    In a habituating network, later timesteps should have fewer spikes
    than early timesteps (the network learns to suppress redundant
    temporal information). This is the within-sample "time flies" effect.

    Returns:
        Dictionary with:
        - early_spike_rate: Average spike rate in first third of timesteps
        - late_spike_rate: Average spike rate in last third of timesteps
        - temporal_reduction: Percentage reduction from early to late
    """
    all_spikes = torch.cat([s for s in spike_recordings], dim=2)
    num_steps = all_spikes.shape[0]

    third = max(num_steps // 3, 1)

    early = all_spikes[:third]
    late = all_spikes[-third:]

    early_rate = early.float().mean().item()
    late_rate = late.float().mean().item()

    reduction = 0.0
    if early_rate > 1e-10:
        reduction = ((early_rate - late_rate) / early_rate) * 100

    return {
        "early_spike_rate": early_rate,
        "late_spike_rate": late_rate,
        "temporal_reduction_pct": reduction,
    }


def compute_per_class_efficiency(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    spike_recordings: list[torch.Tensor],
    num_classes: int = 10,
) -> dict[int, dict[str, float]]:
    """
    Compute efficiency metrics broken down by class.

    Habituation should have differential effects across classes:
    classes with more distinct features (like digit 1) may require
    fewer spikes than classes with overlapping features (like 8 and 9).

    Returns:
        Dictionary mapping class_id to {accuracy, spike_rate, efficiency}.
    """
    total_spikes_per_sample = torch.zeros(predictions.shape[0])
    total_neurons = 0

    for spikes in spike_recordings:
        # spikes: [timesteps, batch, neurons]
        per_sample = spikes.sum(dim=(0, 2))
        total_spikes_per_sample = total_spikes_per_sample + per_sample.cpu()
        total_neurons += spikes.shape[0] * spikes.shape[2]

    predicted = predictions.argmax(dim=1).cpu()
    targets_cpu = targets.cpu()

    results = {}
    for cls in range(num_classes):
        mask = targets_cpu == cls
        if mask.sum() == 0:
            continue

        cls_correct = (predicted[mask] == targets_cpu[mask]).float().mean().item()
        cls_spikes = total_spikes_per_sample[mask].mean().item()
        cls_spike_rate = cls_spikes / max(total_neurons, 1)

        results[cls] = {
            "accuracy": cls_correct,
            "avg_spikes": cls_spikes,
            "spike_rate": cls_spike_rate,
            "efficiency": accuracy_per_spike(cls_correct, cls_spike_rate),
        }

    return results
