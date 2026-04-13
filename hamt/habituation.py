"""
Differentiable habituation module for spiking neural networks.

Implements neural repetition suppression as a drop-in layer that
reduces energy consumption by suppressing responses to familiar
input patterns while maintaining full responsiveness to novelty.
"""

import torch
import torch.nn as nn


class HabituationState:
    """
    Tracks per-neuron familiarity for a single layer.

    Create with HabituationState.initialize(), then pass through
    HabituationLayer.forward() each timestep to get updated state.
    """

    __slots__ = ("familiarity_trace", "input_history", "habituation_strength")

    def __init__(
        self,
        familiarity_trace: torch.Tensor,
        input_history: torch.Tensor,
        habituation_strength: torch.Tensor,
    ):
        self.familiarity_trace = familiarity_trace
        self.input_history = input_history
        self.habituation_strength = habituation_strength

    @staticmethod
    def initialize(
        batch_size: int,
        num_neurons: int,
        history_length: int = 5,
        device: torch.device = torch.device("cpu"),
    ) -> "HabituationState":
        """Create fresh state with no habituation."""
        return HabituationState(
            familiarity_trace=torch.zeros(batch_size, num_neurons, device=device),
            input_history=torch.zeros(
                history_length, batch_size, num_neurons, device=device
            ),
            habituation_strength=torch.zeros(batch_size, num_neurons, device=device),
        )


class HabituationLayer(nn.Module):
    """
    Drop-in habituation layer for any SNN.

    Place between a linear/conv layer and its LIF neuron.
    Learns per-neuron suppression dynamics end-to-end via backprop.

    Args:
        num_neurons: Number of neurons in the layer.
        alpha: Initial max suppression (0-1). 0.5 = up to 50% reduction.
        tau_hab: Habituation rate. Higher = slower buildup of familiarity.
        tau_rec: Recovery rate. Higher = slower recovery from habituation.
        novelty_threshold: Below this similarity, input is considered novel.
        history_length: Timesteps of input history to track.

    Example:
        hab = HabituationLayer(800)
        state = HabituationState.initialize(batch_size, 800, device=device)

        for t in range(num_steps):
            current = linear(spike_input[t])
            current, state = hab(current, state)
            spike, mem = lif(current, mem)
    """

    def __init__(
        self,
        num_neurons: int,
        alpha: float = 0.5,
        tau_hab: float = 0.9,
        tau_rec: float = 0.8,
        novelty_threshold: float = 0.3,
        history_length: int = 5,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.history_length = history_length
        self.novelty_threshold = novelty_threshold

        self._alpha_raw = nn.Parameter(
            torch.full((num_neurons,), _inv_sigmoid(alpha))
        )
        self._tau_hab_raw = nn.Parameter(
            torch.full((num_neurons,), _inv_sigmoid(tau_hab))
        )
        self._tau_rec_raw = nn.Parameter(
            torch.full((num_neurons,), _inv_sigmoid(tau_rec))
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_raw)

    @property
    def tau_hab(self) -> torch.Tensor:
        return torch.sigmoid(self._tau_hab_raw)

    @property
    def tau_rec(self) -> torch.Tensor:
        return torch.sigmoid(self._tau_rec_raw)

    def forward(
        self,
        current_input: torch.Tensor,
        state: HabituationState,
    ) -> tuple[torch.Tensor, HabituationState]:
        """
        Modulate input current based on novelty.

        Familiar patterns are suppressed. Novel patterns pass through.

        Args:
            current_input: Pre-synaptic current [batch, neurons]
            state: Previous HabituationState

        Returns:
            (modulated_current, new_state)
        """
        novelty = self._compute_novelty(current_input, state.input_history)

        is_familiar = (novelty < self.novelty_threshold).float()
        new_familiarity = (
            is_familiar * (self.tau_hab * state.familiarity_trace + (1 - self.tau_hab))
            + (1 - is_familiar) * (self.tau_rec * state.familiarity_trace)
        ).clamp(0.0, 1.0)

        new_strength = self.alpha * new_familiarity
        modulated_current = current_input * (1.0 - new_strength)

        new_history = torch.cat(
            [current_input.unsqueeze(0), state.input_history[:-1]], dim=0
        )

        return modulated_current, HabituationState(
            familiarity_trace=new_familiarity,
            input_history=new_history,
            habituation_strength=new_strength,
        )

    def _compute_novelty(
        self, current: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        mean_history = history.mean(dim=0)
        diff = torch.abs(current - mean_history)
        max_val = torch.max(torch.abs(current), torch.abs(mean_history)).clamp(min=1e-8)
        return diff / max_val

    def get_stats(self) -> dict[str, float]:
        """Return learned parameter means for monitoring."""
        return {
            "alpha": self.alpha.mean().item(),
            "tau_hab": self.tau_hab.mean().item(),
            "tau_rec": self.tau_rec.mean().item(),
        }


def _inv_sigmoid(x: float) -> float:
    x = max(min(x, 0.999), 0.001)
    return -torch.log(torch.tensor(1.0 / x - 1.0)).item()
