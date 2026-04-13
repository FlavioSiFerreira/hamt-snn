"""
Metabolic cost loss function for energy-aware SNN training.

Combines task accuracy with biological energy constraints:
    L = L_task + lambda_e * L_energy + lambda_h * L_habituation
"""

import torch
import torch.nn as nn
from typing import Optional


class MetabolicLoss(nn.Module):
    """
    Drop-in loss replacement for energy-efficient SNN training.

    Replace your CrossEntropyLoss with MetabolicLoss to add
    energy awareness. Compatible with any SNN architecture.

    Args:
        lambda_energy: Energy penalty weight. Start with 0.001, increase
                      to 0.01-0.05 for aggressive energy reduction.
        lambda_habituation: Habituation reward weight. Set to 0 if not
                           using HabituationLayer. Otherwise 0.0005-0.005.
        target_spike_rate: Target sparsity (0.05 = 5%, matching cortex).
                          Lower = more aggressive energy savings.
        ramp_epochs: Epochs to linearly ramp energy penalty from 0 to full.
                    Prevents accuracy collapse in early training.

    Example:
        loss_fn = MetabolicLoss()  # defaults: lambda_energy=0.001, lambda_habituation=0.0005
        loss, info = loss_fn(
            predictions=output_spikes.sum(0),
            targets=labels,
            spike_recordings=[spk1, spk2, spk3],
            habituation_strengths=[hab1, hab2],
            current_epoch=epoch,
        )
        loss.backward()
    """

    def __init__(
        self,
        lambda_energy: float = 0.001,
        lambda_habituation: float = 0.0005,
        target_spike_rate: float = 0.05,
        ramp_epochs: int = 10,
    ):
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_habituation = lambda_habituation
        self.target_spike_rate = target_spike_rate
        self.ramp_epochs = ramp_epochs
        self.task_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        spike_recordings: list[torch.Tensor],
        habituation_strengths: Optional[list[torch.Tensor]] = None,
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined metabolic loss.

        Args:
            predictions: Output spike sums or membrane potentials [batch, classes]
            targets: Class labels [batch]
            spike_recordings: List of spike tensors per layer [steps, batch, neurons]
            habituation_strengths: List of habituation tensors [batch, neurons] per layer.
                                  Pass None if not using HabituationLayer.
            current_epoch: Current epoch for ramp scheduling

        Returns:
            (loss_tensor, info_dict) where info_dict has component breakdowns
        """
        l_task = self.task_loss_fn(predictions, targets)

        ramp = min(1.0, current_epoch / max(self.ramp_epochs, 1))

        l_energy = self._energy_loss(spike_recordings) * ramp

        l_hab = torch.tensor(0.0, device=predictions.device)
        if habituation_strengths is not None and len(habituation_strengths) > 0:
            l_hab = self._habituation_loss(spike_recordings, habituation_strengths) * ramp

        total = l_task + self.lambda_energy * l_energy + self.lambda_habituation * l_hab

        return total, {
            "total": total.item(),
            "task": l_task.item(),
            "energy": l_energy.item(),
            "habituation": l_hab.item(),
            "ramp": ramp,
        }

    def _energy_loss(self, spike_recordings: list[torch.Tensor]) -> torch.Tensor:
        total_spikes = torch.tensor(0.0, device=spike_recordings[0].device)
        total_possible = 0
        for spikes in spike_recordings:
            total_spikes = total_spikes + spikes.sum()
            total_possible += spikes.numel()

        spike_rate = total_spikes / max(total_possible, 1)
        excess = torch.relu(spike_rate - self.target_spike_rate)
        return excess ** 2 + 0.1 * spike_rate

    def _habituation_loss(
        self,
        spike_recordings: list[torch.Tensor],
        habituation_strengths: list[torch.Tensor],
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=spike_recordings[0].device)
        for spikes, strength in zip(spike_recordings, habituation_strengths):
            neuron_rate = spikes.mean(dim=0)
            total = total + (strength * neuron_rate).mean()
        return total
