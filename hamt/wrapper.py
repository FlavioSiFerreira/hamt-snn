"""
High-level wrapper to add HAMT to any existing SNN.

This is the "one function call" integration path. Takes an existing
snnTorch model and returns a HAMT-enhanced version.
"""

import torch
import torch.nn as nn
import snntorch as snn
from typing import Optional

from hamt.habituation import HabituationLayer, HabituationState


class HAMTWrapper(nn.Module):
    """
    Wraps any sequential SNN with habituation layers.

    Automatically inserts HabituationLayer before each spiking neuron
    layer it finds. Works with snnTorch Leaky, Synaptic, and Lapicque
    neuron types.

    Example:
        # Your existing model
        model = nn.Sequential(
            nn.Linear(784, 800),
            snn.Leaky(beta=0.95),
            nn.Linear(800, 10),
            snn.Leaky(beta=0.95, output=True),
        )

        # One-line HAMT enhancement
        hamt_model = HAMTWrapper(model, num_steps=25)

        # Use like normal but get spike recordings + habituation strengths
        output, spike_recs, hab_strengths = hamt_model(spike_input)
    """

    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 25,
        alpha: float = 0.5,
        tau_hab: float = 0.9,
        tau_rec: float = 0.8,
        history_length: int = 5,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.history_length = history_length

        # Parse the model to find linear -> spiking neuron pairs
        self.layers = nn.ModuleList()
        self.hab_layers = nn.ModuleList()
        self.spiking_layers = nn.ModuleList()
        self.layer_types = []  # 'linear', 'hab', 'spiking', 'other'

        modules = list(model.children()) if hasattr(model, 'children') else []
        if isinstance(model, nn.Sequential):
            modules = list(model)

        i = 0
        while i < len(modules):
            mod = modules[i]

            if isinstance(mod, nn.Linear):
                self.layers.append(mod)
                self.layer_types.append("linear")

                # Check if next module is a spiking neuron
                if i + 1 < len(modules) and _is_spiking(modules[i + 1]):
                    out_features = mod.out_features
                    hab = HabituationLayer(
                        num_neurons=out_features,
                        alpha=alpha,
                        tau_hab=tau_hab,
                        tau_rec=tau_rec,
                        history_length=history_length,
                    )
                    self.hab_layers.append(hab)
                    self.layers.append(hab)
                    self.layer_types.append("hab")

                    self.spiking_layers.append(modules[i + 1])
                    self.layers.append(modules[i + 1])
                    self.layer_types.append("spiking")
                    i += 2
                    continue

            elif _is_spiking(mod):
                self.spiking_layers.append(mod)
                self.layers.append(mod)
                self.layer_types.append("spiking")
            else:
                self.layers.append(mod)
                self.layer_types.append("other")

            i += 1

        self._neuron_sizes = []
        for lt, layer in zip(self.layer_types, self.layers):
            if lt == "hab":
                self._neuron_sizes.append(layer.num_neurons)

    def forward(
        self,
        spike_input: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass with automatic habituation.

        Args:
            spike_input: [num_steps, batch, input_features]

        Returns:
            (output_spike_sum, spike_recordings, habituation_strengths)
        """
        batch_size = spike_input.shape[1]
        device = spike_input.device

        # Initialize spiking neuron states
        mem_states = []
        for layer in self.layers:
            if _is_spiking(layer):
                if hasattr(layer, "init_leaky"):
                    mem_states.append(layer.init_leaky())
                elif hasattr(layer, "init_synaptic"):
                    mem_states.append(layer.init_synaptic())
                else:
                    mem_states.append(torch.zeros(batch_size, device=device))

        # Initialize habituation states
        hab_states = []
        for size in self._neuron_sizes:
            hab_states.append(
                HabituationState.initialize(batch_size, size, self.history_length, device)
            )

        spike_recordings = {i: [] for i, lt in enumerate(self.layer_types) if lt == "spiking"}
        final_hab_strengths = []

        spiking_idx = 0
        hab_idx = 0

        for step in range(self.num_steps):
            x = spike_input[step]
            s_idx = 0
            h_idx = 0

            for i, (lt, layer) in enumerate(zip(self.layer_types, self.layers)):
                if lt == "linear" or lt == "other":
                    x = layer(x)
                elif lt == "hab":
                    x, hab_states[h_idx] = layer(x, hab_states[h_idx])
                    h_idx += 1
                elif lt == "spiking":
                    spk, mem_states[s_idx] = layer(x, mem_states[s_idx])
                    spike_recordings[i].append(spk)
                    x = spk
                    s_idx += 1

        # Collect results
        all_spike_recs = []
        for i in sorted(spike_recordings.keys()):
            all_spike_recs.append(torch.stack(spike_recordings[i]))

        for state in hab_states:
            final_hab_strengths.append(state.habituation_strength)

        output_sum = all_spike_recs[-1].sum(dim=0) if all_spike_recs else x

        return output_sum, all_spike_recs, final_hab_strengths


def habituate_snn(
    model: nn.Module,
    num_steps: int = 25,
    alpha: float = 0.5,
    tau_hab: float = 0.9,
    tau_rec: float = 0.8,
) -> HAMTWrapper:
    """
    One-line HAMT enhancement for any sequential SNN.

    Args:
        model: An nn.Sequential SNN with Linear + snnTorch neuron layers
        num_steps: Number of simulation timesteps
        alpha: Max habituation suppression (0.5 = up to 50%)
        tau_hab: Habituation buildup rate
        tau_rec: Habituation recovery rate

    Returns:
        HAMTWrapper that works as a drop-in replacement

    Example:
        model = nn.Sequential(
            nn.Linear(784, 800), snn.Leaky(beta=0.95),
            nn.Linear(800, 10), snn.Leaky(beta=0.95, output=True),
        )
        hamt_model = habituate_snn(model, num_steps=25)
        output, spikes, hab = hamt_model(spike_input)
    """
    return HAMTWrapper(
        model, num_steps=num_steps, alpha=alpha,
        tau_hab=tau_hab, tau_rec=tau_rec,
    )


def _is_spiking(module: nn.Module) -> bool:
    """Check if a module is a snnTorch spiking neuron."""
    spiking_types = (snn.Leaky, snn.Synaptic, snn.Lapicque)
    try:
        # Also check for RLeaky, RSynaptic if available
        spiking_types = spiking_types + (snn.RLeaky, snn.RSynaptic)
    except AttributeError:
        pass
    return isinstance(module, spiking_types)
