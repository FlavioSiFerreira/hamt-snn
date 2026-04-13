"""
Persistent Habituation Module: Cross-Sample Familiarity Tracking.

Unlike the original HabituationLayer which resets familiarity for each
new input sample, this module maintains a persistent memory of neuron
activation patterns across the entire training set. Neurons that
consistently fire for common patterns accumulate long-term familiarity
and get progressively suppressed.

Biological basis:
    Short-term habituation (seconds): synaptic vesicle depletion.
    Long-term habituation (days/weeks): structural changes in synapses,
    reduced receptor expression, altered gene transcription.

    The original module only captures short-term habituation (within one
    forward pass). This module adds long-term habituation (across the
    full training dataset), matching how biological systems actually
    allocate metabolic resources.

Key difference:
    Original: "I've seen this spike pattern for the last 5 timesteps"
    Persistent: "I've seen this activation pattern thousands of times
    across training. This neuron always fires for class 3. Suppress it."
"""

import torch
import torch.nn as nn
from typing import Optional

from src.habituation.habituation_module import HabituationState


class PersistentHabituationLayer(nn.Module):
    """
    Habituation layer with persistent cross-sample memory.

    Drop-in replacement for HabituationLayer. Same forward signature,
    same HabituationState, but adds a persistent EMA buffer that tracks
    neuron activation patterns across training batches.

    The total familiarity is a weighted combination of:
    - Short-term: within-sample novelty (same as original)
    - Long-term: cross-sample novelty (new, persistent)

    Parameters:
        num_neurons: Number of neurons in the layer
        alpha_init: Maximum suppression strength (learnable, per-neuron)
        tau_hab: Short-term habituation time constant (learnable)
        tau_rec: Short-term recovery time constant (learnable)
        novelty_threshold: Threshold for short-term novelty detection
        history_length: Timesteps of short-term history
        memory_decay: EMA decay for long-term memory (0.999 = slow, 0.99 = fast)
        long_term_weight: How much long-term familiarity contributes (0-1)
        warmup_batches: Number of batches before long-term memory activates
    """

    def __init__(
        self,
        num_neurons: int,
        alpha_init: float = 0.5,
        tau_hab: float = 0.9,
        tau_rec: float = 0.8,
        novelty_threshold: float = 0.3,
        history_length: int = 5,
        memory_decay: float = 0.999,
        long_term_weight: float = 0.3,
        warmup_batches: int = 100,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.history_length = history_length
        self.novelty_threshold = novelty_threshold
        self.memory_decay = memory_decay
        self.long_term_weight = long_term_weight
        self.warmup_batches = warmup_batches

        # Learnable parameters (same as original)
        self._alpha_raw = nn.Parameter(
            torch.full((num_neurons,), self._inverse_sigmoid(alpha_init))
        )
        self._tau_hab_raw = nn.Parameter(
            torch.full((num_neurons,), self._inverse_sigmoid(tau_hab))
        )
        self._tau_rec_raw = nn.Parameter(
            torch.full((num_neurons,), self._inverse_sigmoid(tau_rec))
        )

        # Persistent buffers (survive across batches, not trainable)
        # EMA of mean activation per neuron across all training samples
        self.register_buffer(
            "activation_memory", torch.zeros(num_neurons)
        )
        # EMA of activation variance per neuron (for novelty scaling)
        self.register_buffer(
            "activation_var", torch.ones(num_neurons)
        )
        # Count of batches seen (for warmup)
        self.register_buffer(
            "batch_count", torch.tensor(0, dtype=torch.long)
        )

    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        x = max(min(x, 0.999), 0.001)
        return -torch.log(torch.tensor(1.0 / x - 1.0)).item()

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_raw)

    @property
    def tau_hab(self) -> torch.Tensor:
        return torch.sigmoid(self._tau_hab_raw)

    @property
    def tau_rec(self) -> torch.Tensor:
        return torch.sigmoid(self._tau_rec_raw)

    @property
    def long_term_active(self) -> bool:
        """Long-term memory only activates after warmup."""
        return self.batch_count.item() >= self.warmup_batches

    def forward(
        self,
        current_input: torch.Tensor,
        state: HabituationState,
    ) -> tuple[torch.Tensor, HabituationState]:
        """
        Apply habituation with both short-term and long-term familiarity.

        Same signature as HabituationLayer.forward() for drop-in compatibility.

        Args:
            current_input: Pre-synaptic current [batch, neurons]
            state: Previous (short-term) habituation state

        Returns:
            (modulated_current, new_state)
        """
        # === Short-term familiarity (within-sample, same as original) ===
        short_term_novelty = self._compute_short_term_novelty(
            current_input, state.input_history
        )
        is_familiar_st = (short_term_novelty < self.novelty_threshold).float()

        new_familiarity_st = (
            is_familiar_st * (self.tau_hab * state.familiarity_trace + (1 - self.tau_hab))
            + (1 - is_familiar_st) * (self.tau_rec * state.familiarity_trace)
        ).clamp(0.0, 1.0)

        # === Long-term familiarity (cross-sample, new) ===
        if self.long_term_active:
            long_term_familiarity = self._compute_long_term_familiarity(
                current_input
            )
            # Combine short-term and long-term
            combined_familiarity = (
                (1 - self.long_term_weight) * new_familiarity_st
                + self.long_term_weight * long_term_familiarity
            )
        else:
            combined_familiarity = new_familiarity_st

        # === Update persistent memory (only during training) ===
        if self.training:
            self._update_memory(current_input)

        # === Modulate current ===
        new_strength = self.alpha * combined_familiarity
        modulated_current = current_input * (1.0 - new_strength)

        # === Update short-term history ===
        new_history = torch.cat(
            [current_input.unsqueeze(0), state.input_history[:-1]],
            dim=0,
        )

        new_state = HabituationState(
            familiarity_trace=new_familiarity_st,
            input_history=new_history,
            habituation_strength=new_strength,
        )

        return modulated_current, new_state

    def _compute_short_term_novelty(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """Same as original: normalized difference from recent timesteps."""
        mean_history = history.mean(dim=0)
        diff = torch.abs(current - mean_history)
        max_val = torch.max(
            torch.abs(current), torch.abs(mean_history)
        ).clamp(min=1e-8)
        return diff / max_val

    def _compute_long_term_familiarity(
        self,
        current_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute how familiar the current activation pattern is
        relative to the long-term memory of typical activations.

        Returns values in [0, 1] where 1 = very familiar (close to
        the running mean), 0 = novel (far from running mean).
        """
        # Z-score of current activation relative to long-term stats
        deviation = torch.abs(current_input - self.activation_memory.unsqueeze(0))
        std = torch.sqrt(self.activation_var.unsqueeze(0)).clamp(min=1e-8)
        z_score = deviation / std

        # Convert z-score to familiarity: low z = familiar, high z = novel
        # Using sigmoid: z=0 -> familiarity=1, z=2 -> familiarity~0.12
        familiarity = torch.sigmoid(2.0 - z_score)

        return familiarity

    @torch.no_grad()
    def _update_memory(self, current_input: torch.Tensor):
        """
        Update the persistent EMA of activation patterns.

        Called once per forward pass during training. Uses the batch
        mean to update the running statistics.
        """
        batch_mean = current_input.mean(dim=0)
        batch_var = current_input.var(dim=0)

        if self.batch_count == 0:
            # First batch: initialize directly
            self.activation_memory.copy_(batch_mean)
            self.activation_var.copy_(batch_var.clamp(min=1e-8))
        else:
            # EMA update
            d = self.memory_decay
            self.activation_memory.mul_(d).add_(batch_mean, alpha=1 - d)
            self.activation_var.mul_(d).add_(batch_var.clamp(min=1e-8), alpha=1 - d)

        self.batch_count += 1

    def get_memory_stats(self) -> dict:
        """Return diagnostic info about the persistent memory state."""
        return {
            "batch_count": self.batch_count.item(),
            "long_term_active": self.long_term_active,
            "mean_activation": self.activation_memory.mean().item(),
            "mean_variance": self.activation_var.mean().item(),
            "memory_decay": self.memory_decay,
            "long_term_weight": self.long_term_weight,
        }
