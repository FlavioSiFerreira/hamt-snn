"""
Energy and efficiency metrics for HAMT evaluation.
"""

import torch

# Loihi 2 approximate energy costs
ENERGY_PER_SPIKE_PJ = 120.0
ENERGY_PER_SYNOP_PJ = 23.0
PJ_TO_JOULES = 1e-12


def compute_spike_rate(spike_recordings: list[torch.Tensor]) -> float:
    """Average fraction of neurons firing per timestep (0.0 to 1.0)."""
    total = sum(s.sum().item() for s in spike_recordings)
    possible = sum(s.numel() for s in spike_recordings)
    return total / max(possible, 1)


def estimate_energy(
    spike_recordings: list[torch.Tensor],
    fan_outs: list[int],
) -> float:
    """Estimated energy in joules using Loihi 2 cost model."""
    total_spikes = sum(int(s.sum().item()) for s in spike_recordings)
    syn_ops = sum(
        int(s.sum().item()) * fo
        for s, fo in zip(spike_recordings, fan_outs)
    )
    energy_pj = total_spikes * ENERGY_PER_SPIKE_PJ + syn_ops * ENERGY_PER_SYNOP_PJ
    return energy_pj * PJ_TO_JOULES


def accuracy_per_joule(accuracy: float, energy_joules: float) -> float:
    """Accuracy achieved per joule of estimated energy. Higher is better."""
    if energy_joules < 1e-20:
        return float("inf") if accuracy > 0 else 0.0
    return accuracy / energy_joules
