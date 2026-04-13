"""
HAMT: Habituation-Aware Metabolic Training for spiking neural networks.

Drop-in habituation and metabolic-cost modules for SNNs.

Observed behavior on pattern recognition tasks (MNIST, Fashion-MNIST, EMNIST):
around 20 to 40% reduction in estimated inference energy with negligible or
mildly positive accuracy change. Reduction does not compound with longer
training and disappears on natural image datasets (CIFAR-10). Ablation shows
the habituation term is the active ingredient; the metabolic penalty alone is
negligible. See README for full results.

Quick start:
    from hamt import HabituationLayer, MetabolicLoss

    hab = HabituationLayer(num_neurons=800)
    modulated_current, state = hab(current, state)

    loss_fn = MetabolicLoss()  # defaults: lambda_energy=0.001, lambda_habituation=0.0005
    loss, info = loss_fn(predictions, targets, spike_recordings, hab_strengths)
"""

__version__ = "0.1.0"

from hamt.habituation import HabituationLayer, HabituationState
from hamt.loss import MetabolicLoss
from hamt.wrapper import habituate_snn
from hamt.metrics import compute_spike_rate, estimate_energy, accuracy_per_joule

__all__ = [
    "HabituationLayer",
    "HabituationState",
    "MetabolicLoss",
    "habituate_snn",
    "compute_spike_rate",
    "estimate_energy",
    "accuracy_per_joule",
]


def habituate_norse(model, num_steps=25, **kwargs):
    """One-line HAMT for Norse framework SNNs."""
    from hamt.norse_compat import habituate_norse as _fn
    return _fn(model, num_steps, **kwargs)


def get_lava_process():
    """Get Lava HabituationProcess for Intel Loihi deployment."""
    from hamt.lava_compat import HabituationProcess
    return HabituationProcess
