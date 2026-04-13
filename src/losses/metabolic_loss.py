"""
Metabolic Cost Loss Function: The dual-objective training signal.

Biological basis:
    The brain does not optimize for accuracy alone. Every spike costs
    ATP (adenosine triphosphate), and the brain's total energy budget
    is tightly constrained (~20W). Evolution has selected for neural
    circuits that achieve sufficient accuracy at minimal metabolic cost.

    This loss function replicates that dual objective:
        L_total = L_task + lambda_energy * L_energy + lambda_hab * L_habituation

    Where:
        L_task: Standard cross-entropy for classification accuracy
        L_energy: Penalizes total spike count and synaptic operations
        L_habituation: Rewards progressive spike reduction for repeated patterns
                       (the habituation bonus, incentivizing the network to
                        learn which patterns are "familiar" and suppress them)
"""

import torch
import torch.nn as nn
from typing import Optional


class MetabolicLoss(nn.Module):
    """
    Combined loss: task accuracy + metabolic energy cost + habituation reward.

    The habituation component is the novel element. Standard spike-rate
    regularization just penalizes all spikes equally. The habituation
    reward specifically incentivizes the network to fire less for
    patterns it has seen before, while maintaining full response to
    novel inputs. This mirrors the brain's strategy.
    """

    def __init__(
        self,
        lambda_energy: float = 0.001,
        lambda_habituation: float = 0.0005,
        target_spike_rate: float = 0.05,
        ramp_epochs: int = 10,
    ):
        """
        Args:
            lambda_energy: Weight for energy penalty term.
                          Higher = more aggressive spike reduction.
            lambda_habituation: Weight for habituation reward term.
                               Higher = more reward for temporal efficiency.
            target_spike_rate: Target average spike rate (5% matches cortex).
            ramp_epochs: Number of epochs to linearly ramp up energy penalty.
                        Prevents collapse in early training when accuracy
                        is still being learned.
        """
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
        Compute the combined metabolic loss.

        Args:
            predictions: Model output [batch, num_classes]
            targets: Ground truth labels [batch]
            spike_recordings: List of spike tensors per layer
                             [num_steps, batch, neurons] each
            habituation_strengths: Optional list of habituation strength
                                  tensors per layer [batch, neurons] each.
                                  If None, habituation term is skipped.
            current_epoch: Current training epoch for lambda ramping

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # 1. Task loss (cross-entropy)
        l_task = self.task_loss_fn(predictions, targets)

        # 2. Energy loss (spike rate penalty)
        # Ramp up gradually to let accuracy establish first
        ramp_factor = min(1.0, current_epoch / max(self.ramp_epochs, 1))
        l_energy = self._compute_energy_loss(spike_recordings) * ramp_factor

        # 3. Habituation reward (novel component)
        l_habituation = torch.tensor(0.0, device=predictions.device)
        if habituation_strengths is not None and len(habituation_strengths) > 0:
            l_habituation = self._compute_habituation_loss(
                spike_recordings, habituation_strengths
            ) * ramp_factor

        # Combined loss
        total = (
            l_task
            + self.lambda_energy * l_energy
            + self.lambda_habituation * l_habituation
        )

        components = {
            "task": l_task.item(),
            "energy": l_energy.item(),
            "habituation": l_habituation.item(),
            "total": total.item(),
            "ramp_factor": ramp_factor,
        }

        return total, components

    def _compute_energy_loss(
        self,
        spike_recordings: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Penalize spike rates that exceed the biological target.

        Uses a soft penalty that increases quadratically as spike rate
        exceeds the target. Below-target rates are not penalized
        (the network is already efficient enough).
        """
        total_spikes = torch.tensor(0.0, device=spike_recordings[0].device)
        total_possible = 0

        for spikes in spike_recordings:
            total_spikes = total_spikes + spikes.sum()
            total_possible += spikes.numel()

        spike_rate = total_spikes / max(total_possible, 1)

        # Quadratic penalty for exceeding target rate
        excess = torch.relu(spike_rate - self.target_spike_rate)
        energy_loss = excess ** 2

        # Also add a small linear term to always push toward efficiency
        energy_loss = energy_loss + 0.1 * spike_rate

        return energy_loss

    def _compute_habituation_loss(
        self,
        spike_recordings: list[torch.Tensor],
        habituation_strengths: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Reward the network for spiking less on habituated (familiar) inputs.

        The key insight: standard spike regularization penalizes ALL spikes
        equally. This term specifically rewards correlation between
        habituation strength and spike suppression. A neuron that is highly
        habituated (familiar input) but still firing is penalized more than
        a neuron that is not habituated (novel input) and firing.

        This encourages the network to allocate its spike budget efficiently:
        spend spikes on novel/surprising inputs, save them on familiar ones.

        L_hab = mean(habituation_strength * spike_rate_per_neuron)

        When habituation is high and spikes are high -> large loss (bad)
        When habituation is high and spikes are low -> small loss (good)
        When habituation is low and spikes are high -> small loss (fine, it's novel)
        """
        total_hab_loss = torch.tensor(0.0, device=spike_recordings[0].device)

        for spikes, hab_strength in zip(spike_recordings, habituation_strengths):
            # Average spike rate per neuron across timesteps: [batch, neurons]
            neuron_spike_rate = spikes.mean(dim=0)

            # Penalize spikes that occur despite high habituation
            # This is the core novelty: energy should go to novel inputs
            hab_spike_product = hab_strength * neuron_spike_rate
            total_hab_loss = total_hab_loss + hab_spike_product.mean()

        return total_hab_loss
