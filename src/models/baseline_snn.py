"""
Baseline SNN model for comparison against HAMT.

Standard feedforward SNN with Leaky Integrate-and-Fire (LIF) neurons,
trained with surrogate gradient descent (the current dominant approach).
No energy awareness, no habituation. This is what HAMT must beat.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class BaselineSNN(nn.Module):
    """
    Standard feedforward SNN for MNIST classification.

    Architecture: 784 -> 800 -> 800 -> 10
    Neuron model: Leaky Integrate-and-Fire (LIF)
    Training: Surrogate gradient (fast sigmoid)
    Readout: Membrane potential of output layer at final timestep

    This serves as the control condition. HAMT will use the same
    architecture but with habituation and metabolic cost training.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 800,
        output_size: int = 10,
        num_steps: int = 25,
        beta: float = 0.95,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Surrogate gradient function for backprop through spikes
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Layer 1: Input -> Hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            learn_beta=True,
        )

        # Layer 2: Hidden -> Hidden
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            learn_beta=True,
        )

        # Layer 3: Hidden -> Output
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lif3 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            learn_beta=True,
            output=True,
        )

        # Fan-outs for energy calculation (connections per neuron to next layer)
        self.fan_outs = [hidden_size, hidden_size, output_size]

    def forward(
        self,
        spike_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through the SNN.

        Args:
            spike_input: Spike train tensor [num_steps, batch, input_size]

        Returns:
            Tuple of:
                - output_membrane: Final membrane potentials [batch, output_size]
                - output_spikes_sum: Summed output spikes [batch, output_size]
                - spike_recordings: List of spike tensors per layer for metrics
        """
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Spike recordings for energy metrics
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []

        for step in range(self.num_steps):
            x = spike_input[step]

            # Layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3 (output)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)

        # Stack recordings: [num_steps, batch, neurons]
        spk1_rec = torch.stack(spk1_rec)
        spk2_rec = torch.stack(spk2_rec)
        spk3_rec = torch.stack(spk3_rec)

        # Output: sum of output spikes over time (rate coding readout)
        output_spikes_sum = spk3_rec.sum(dim=0)

        return mem3, output_spikes_sum, [spk1_rec, spk2_rec, spk3_rec]

    def get_fan_outs(self) -> list[int]:
        """Return fan-out sizes for energy estimation."""
        return self.fan_outs
