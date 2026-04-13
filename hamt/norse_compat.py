"""
HAMT compatibility layer for Norse framework.

Norse is PyTorch-based, so the core HabituationLayer works directly.
This module provides a convenience wrapper that detects Norse neuron
types and auto-inserts habituation layers.

NOTE: Requires norse to be installed: pip install norse
"""

import torch
import torch.nn as nn
from hamt.habituation import HabituationLayer, HabituationState

try:
    import norse.torch as norse
    NORSE_AVAILABLE = True
except ImportError:
    NORSE_AVAILABLE = False


def _is_norse_spiking(module: nn.Module) -> bool:
    """Check if a module is a Norse spiking neuron."""
    if not NORSE_AVAILABLE:
        return False
    norse_types = (
        norse.LIFCell,
        norse.LIFRecurrentCell,
        norse.LICell,
    )
    try:
        norse_types = norse_types + (norse.LIFAdExCell, norse.LIFExCell)
    except AttributeError:
        pass
    return isinstance(module, norse_types)


class NorseHAMTWrapper(nn.Module):
    """
    Wraps a Norse sequential SNN with HAMT habituation layers.

    Works like the snnTorch wrapper but detects Norse neuron types.
    Norse neurons use (input, state) -> (output, state) calling convention.

    Example:
        import norse.torch as norse

        model = nn.Sequential(
            nn.Linear(784, 800),
            norse.LIFCell(),
            nn.Linear(800, 10),
            norse.LIFCell(),
        )

        hamt_model = habituate_norse(model, num_steps=25)
        output, spikes, hab = hamt_model(spike_input)
    """

    def __init__(self, model, num_steps=25, alpha=0.5,
                 tau_hab=0.9, tau_rec=0.8, history_length=5):
        super().__init__()
        self.num_steps = num_steps
        self.history_length = history_length

        self.layers = nn.ModuleList()
        self.layer_types = []

        modules = list(model) if isinstance(model, nn.Sequential) else list(model.children())
        i = 0
        while i < len(modules):
            mod = modules[i]
            if isinstance(mod, nn.Linear):
                self.layers.append(mod)
                self.layer_types.append("linear")
                if i + 1 < len(modules) and _is_norse_spiking(modules[i + 1]):
                    out_features = mod.out_features
                    hab = HabituationLayer(out_features, alpha, tau_hab, tau_rec,
                                          history_length=history_length)
                    self.layers.append(hab)
                    self.layer_types.append("hab")
                    self.layers.append(modules[i + 1])
                    self.layer_types.append("spiking")
                    i += 2
                    continue
            elif _is_norse_spiking(mod):
                self.layers.append(mod)
                self.layer_types.append("spiking")
            else:
                self.layers.append(mod)
                self.layer_types.append("other")
            i += 1

        self._neuron_sizes = [
            layer.num_neurons for lt, layer in zip(self.layer_types, self.layers)
            if lt == "hab"
        ]

    def forward(self, spike_input):
        batch_size = spike_input.shape[1]
        device = spike_input.device

        neuron_states = [None] * sum(1 for lt in self.layer_types if lt == "spiking")
        hab_states = [
            HabituationState.initialize(batch_size, size, self.history_length, device)
            for size in self._neuron_sizes
        ]

        spike_recordings = {i: [] for i, lt in enumerate(self.layer_types) if lt == "spiking"}

        for step in range(self.num_steps):
            x = spike_input[step]
            s_idx = 0
            h_idx = 0

            for i, (lt, layer) in enumerate(zip(self.layer_types, self.layers)):
                if lt in ("linear", "other"):
                    x = layer(x)
                elif lt == "hab":
                    x, hab_states[h_idx] = layer(x, hab_states[h_idx])
                    h_idx += 1
                elif lt == "spiking":
                    x, neuron_states[s_idx] = layer(x, neuron_states[s_idx])
                    spike_recordings[i].append(x)
                    s_idx += 1

        all_spike_recs = [
            torch.stack(spike_recordings[i]) for i in sorted(spike_recordings.keys())
        ]
        final_hab = [s.habituation_strength for s in hab_states]
        output_sum = all_spike_recs[-1].sum(dim=0) if all_spike_recs else x

        return output_sum, all_spike_recs, final_hab


def habituate_norse(model, num_steps=25, alpha=0.5, tau_hab=0.9, tau_rec=0.8):
    """
    One-line HAMT enhancement for Norse sequential SNNs.

    Args:
        model: nn.Sequential with Linear + Norse neuron layers
        num_steps: Simulation timesteps

    Returns:
        NorseHAMTWrapper
    """
    if not NORSE_AVAILABLE:
        raise ImportError("Norse not installed. pip install norse")
    return NorseHAMTWrapper(model, num_steps, alpha, tau_hab, tau_rec)
