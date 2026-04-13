"""
Habituation Module: The core biological innovation.

Implements neural habituation (repetition suppression) as a differentiable
module that can be inserted into any SNN layer. When a neuron sees the same
input pattern repeatedly, its response is progressively suppressed, saving
energy. Novel patterns receive full processing.

Biological basis:
    In biological neurons, repeated stimulation leads to decreased synaptic
    vesicle release (short-term synaptic depression) and reduced postsynaptic
    response. This is one of the brain's primary energy-saving mechanisms.
    Only surprising (novel) information gets full metabolic investment.

    This is also why subjective time perception accelerates with age:
    familiar environments and routines trigger less neural processing,
    creating fewer salient memory markers.

Implementation:
    Each neuron maintains a "familiarity trace" that tracks how similar
    current input is to recent inputs. High familiarity reduces the
    neuron's gain (effective threshold increase), suppressing spikes
    for predictable inputs while preserving full responsiveness to
    novel stimuli.
"""

import torch
import torch.nn as nn
from typing import Optional


class HabituationState:
    """
    Immutable snapshot of habituation state for a single layer.

    Rather than mutating state in place, each timestep produces
    a new HabituationState from the previous one.
    """

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
        """Create initial (no habituation) state."""
        return HabituationState(
            familiarity_trace=torch.zeros(batch_size, num_neurons, device=device),
            input_history=torch.zeros(
                history_length, batch_size, num_neurons, device=device
            ),
            habituation_strength=torch.zeros(batch_size, num_neurons, device=device),
        )


class HabituationLayer(nn.Module):
    """
    Differentiable habituation layer for SNN neurons.

    Placed before each LIF neuron layer, it modulates the input current
    based on novelty. Familiar patterns get attenuated, novel patterns
    pass through at full strength.

    The key equation:
        modulated_current = current * (1 - alpha * habituation_strength)

    Where habituation_strength increases when current input matches
    recent history and decays when inputs change.

    Parameters:
        alpha_init: Initial maximum suppression factor (learnable).
                    Controls how much familiar inputs are attenuated.
                    Biological range: 0.3 to 0.8 (30-80% suppression).

        tau_hab: Habituation time constant (learnable).
                 Controls how quickly familiarity builds.
                 Larger = slower habituation (more conservative).

        tau_rec: Recovery time constant (learnable).
                 Controls how quickly habituation dissipates for novel inputs.
                 Larger = slower recovery.

        novelty_threshold: Similarity threshold below which input is
                          considered "novel" and habituation resets.
    """

    def __init__(
        self,
        num_neurons: int,
        alpha_init: float = 0.5,
        tau_hab: float = 0.9,
        tau_rec: float = 0.8,
        novelty_threshold: float = 0.3,
        history_length: int = 5,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.history_length = history_length
        self.novelty_threshold = novelty_threshold

        # Learnable parameters (per-neuron for fine-grained control)
        # Using sigmoid-constrained parameters to keep values in valid ranges
        self._alpha_raw = nn.Parameter(
            torch.full((num_neurons,), self._inverse_sigmoid(alpha_init))
        )
        self._tau_hab_raw = nn.Parameter(
            torch.full((num_neurons,), self._inverse_sigmoid(tau_hab))
        )
        self._tau_rec_raw = nn.Parameter(
            torch.full((num_neurons,), self._inverse_sigmoid(tau_rec))
        )

    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        """Inverse of sigmoid for initialization."""
        x = max(min(x, 0.999), 0.001)
        return -torch.log(torch.tensor(1.0 / x - 1.0)).item()

    @property
    def alpha(self) -> torch.Tensor:
        """Maximum suppression factor, constrained to (0, 1)."""
        return torch.sigmoid(self._alpha_raw)

    @property
    def tau_hab(self) -> torch.Tensor:
        """Habituation time constant, constrained to (0, 1)."""
        return torch.sigmoid(self._tau_hab_raw)

    @property
    def tau_rec(self) -> torch.Tensor:
        """Recovery time constant, constrained to (0, 1)."""
        return torch.sigmoid(self._tau_rec_raw)

    def forward(
        self,
        current_input: torch.Tensor,
        state: HabituationState,
    ) -> tuple[torch.Tensor, HabituationState]:
        """
        Apply habituation modulation to input current.

        Args:
            current_input: Pre-synaptic current [batch, neurons]
            state: Previous habituation state

        Returns:
            Tuple of:
                - modulated_current: Attenuated current [batch, neurons]
                - new_state: Updated habituation state (new object, no mutation)
        """
        # Step 1: Compute novelty score
        # Compare current input to recent history using cosine similarity
        novelty = self._compute_novelty(current_input, state.input_history)

        # Step 2: Update familiarity trace
        # High novelty -> trace decays (recovery)
        # Low novelty -> trace grows (habituation)
        is_familiar = (novelty < self.novelty_threshold).float()

        new_familiarity = (
            is_familiar * (self.tau_hab * state.familiarity_trace + (1 - self.tau_hab))
            + (1 - is_familiar) * (self.tau_rec * state.familiarity_trace)
        )
        new_familiarity = new_familiarity.clamp(0.0, 1.0)

        # Step 3: Compute habituation strength from familiarity
        new_strength = self.alpha * new_familiarity

        # Step 4: Modulate current (the core operation)
        # Familiar inputs are suppressed, novel inputs pass through
        gain = 1.0 - new_strength
        modulated_current = current_input * gain

        # Step 5: Update input history (shift and insert, no in-place mutation)
        new_history = torch.cat(
            [current_input.unsqueeze(0), state.input_history[:-1]],
            dim=0,
        )

        new_state = HabituationState(
            familiarity_trace=new_familiarity,
            input_history=new_history,
            habituation_strength=new_strength,
        )

        return modulated_current, new_state

    def _compute_novelty(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-neuron novelty score by comparing current input
        to the mean of recent inputs.

        Returns values in [0, 1] where 1 = completely novel, 0 = identical.
        """
        # Mean of recent inputs
        mean_history = history.mean(dim=0)

        # Normalized absolute difference (per neuron)
        diff = torch.abs(current - mean_history)
        max_val = torch.max(
            torch.abs(current), torch.abs(mean_history)
        ).clamp(min=1e-8)
        novelty = diff / max_val

        return novelty
