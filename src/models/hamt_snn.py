"""
HAMT-SNN: Habituation-Aware Metabolic Training Spiking Neural Network.

This model integrates habituation layers between each linear+LIF pair.
Same architecture as BaselineSNN (784->800->800->10) but with
HabituationLayer modules that modulate input currents based on
pattern familiarity.

The habituation parameters (alpha, tau_hab, tau_rec) are learnable
and trained end-to-end alongside the network weights. This means
the network learns both WHAT to compute and WHEN to save energy.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from ..habituation.habituation_module import HabituationLayer, HabituationState


class HAMTSNN(nn.Module):
    """
    SNN with integrated habituation for energy-efficient inference.

    Identical architecture to BaselineSNN but with HabituationLayer
    inserted before each LIF neuron. The habituation layers learn to
    suppress responses to familiar patterns, reducing total spike count
    while preserving accuracy on novel/important inputs.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 800,
        output_size: int = 10,
        num_steps: int = 25,
        beta: float = 0.95,
        hab_alpha: float = 0.5,
        hab_tau: float = 0.9,
        hab_recovery: float = 0.8,
        history_length: int = 5,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.history_length = history_length

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Layer 1: Input -> Hidden (with habituation)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hab1 = HabituationLayer(
            num_neurons=hidden_size,
            alpha_init=hab_alpha,
            tau_hab=hab_tau,
            tau_rec=hab_recovery,
            history_length=history_length,
        )
        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            learn_beta=True,
        )

        # Layer 2: Hidden -> Hidden (with habituation)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.hab2 = HabituationLayer(
            num_neurons=hidden_size,
            alpha_init=hab_alpha,
            tau_hab=hab_tau,
            tau_rec=hab_recovery,
            history_length=history_length,
        )
        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            learn_beta=True,
        )

        # Layer 3: Hidden -> Output (no habituation on output layer,
        # because output spikes carry the classification signal)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lif3 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            learn_beta=True,
            output=True,
        )

        self.fan_outs = [hidden_size, hidden_size, output_size]

    def forward(
        self,
        spike_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass with habituation-modulated processing.

        Args:
            spike_input: Spike train tensor [num_steps, batch, input_size]

        Returns:
            Tuple of:
                - output_membrane: Final membrane potentials [batch, output_size]
                - output_spikes_sum: Summed output spikes [batch, output_size]
                - spike_recordings: List of spike tensors per layer
                - habituation_strengths: List of final habituation strengths per layer
        """
        batch_size = spike_input.shape[1]
        device = spike_input.device

        # Initialize LIF states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Initialize habituation states
        hab_state1 = HabituationState.initialize(
            batch_size, self.hidden_size, self.history_length, device
        )
        hab_state2 = HabituationState.initialize(
            batch_size, self.hidden_size, self.history_length, device
        )

        # Spike recordings
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []

        for step in range(self.num_steps):
            x = spike_input[step]

            # Layer 1: Linear -> Habituation -> LIF
            cur1 = self.fc1(x)
            cur1_hab, hab_state1 = self.hab1(cur1, hab_state1)
            spk1, mem1 = self.lif1(cur1_hab, mem1)

            # Layer 2: Linear -> Habituation -> LIF
            cur2 = self.fc2(spk1)
            cur2_hab, hab_state2 = self.hab2(cur2, hab_state2)
            spk2, mem2 = self.lif2(cur2_hab, mem2)

            # Layer 3: Linear -> LIF (no habituation on readout)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)

        # Stack recordings
        spk1_rec = torch.stack(spk1_rec)
        spk2_rec = torch.stack(spk2_rec)
        spk3_rec = torch.stack(spk3_rec)

        output_spikes_sum = spk3_rec.sum(dim=0)

        spike_recordings = [spk1_rec, spk2_rec, spk3_rec]
        habituation_strengths = [
            hab_state1.habituation_strength,
            hab_state2.habituation_strength,
        ]

        return mem3, output_spikes_sum, spike_recordings, habituation_strengths

    def get_fan_outs(self) -> list[int]:
        """Return fan-out sizes for energy estimation."""
        return self.fan_outs

    def get_habituation_stats(self) -> dict[str, float]:
        """
        Return current learned habituation parameters for analysis.

        Useful for understanding what the network learned about
        when to suppress vs. when to respond.
        """
        return {
            "layer1_alpha_mean": self.hab1.alpha.mean().item(),
            "layer1_tau_hab_mean": self.hab1.tau_hab.mean().item(),
            "layer1_tau_rec_mean": self.hab1.tau_rec.mean().item(),
            "layer2_alpha_mean": self.hab2.alpha.mean().item(),
            "layer2_tau_hab_mean": self.hab2.tau_hab.mean().item(),
            "layer2_tau_rec_mean": self.hab2.tau_rec.mean().item(),
        }
