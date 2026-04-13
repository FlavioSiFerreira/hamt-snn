"""
HAMT compatibility layer for Intel Lava framework.

Lava uses a Process-based architecture where neurons are stateful
computational objects. This module provides a HabituationProcess
that can be inserted into Lava SNN pipelines, and a utility to
convert HAMT-trained PyTorch models to Lava deployable format.

NOTE: This module requires lava-nc to be installed.
      pip install lava-nc

The core habituation logic is identical to the PyTorch version.
Only the interface changes to match Lava's Process API.
"""

try:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.core.process.ports.ports import InPort, OutPort
    from lava.magma.core.process.variable import Var
    from lava.magma.core.model.py.model import PyLoihiProcessModel
    from lava.magma.core.model.py.ports import PyInPort, PyOutPort
    from lava.magma.core.model.py.type import LavaPyType
    from lava.magma.core.resources import CPU
    from lava.magma.core.decorator import implements, requires
    import numpy as np
    LAVA_AVAILABLE = True
except ImportError:
    LAVA_AVAILABLE = False


def check_lava():
    """Check if Lava is available."""
    if not LAVA_AVAILABLE:
        raise ImportError(
            "Intel Lava framework not found. Install with: pip install lava-nc"
        )


if LAVA_AVAILABLE:

    class HabituationProcess(AbstractProcess):
        """
        Lava Process implementing HAMT habituation.

        Receives pre-synaptic current, modulates it based on
        input familiarity, and outputs the attenuated current.
        Connects between a Dense (synapse) process and a LIF process.

        Parameters:
            shape: Neuron shape tuple
            alpha: Maximum suppression strength (0-1)
            tau_hab: Habituation time constant (0-1)
            tau_rec: Recovery time constant (0-1)
            novelty_threshold: Threshold for novel vs familiar classification
            history_length: Number of recent inputs to track
        """

        def __init__(self, shape, alpha=0.5, tau_hab=0.9, tau_rec=0.8,
                     novelty_threshold=0.3, history_length=5, **kwargs):
            super().__init__(shape=shape, **kwargs)
            n = np.prod(shape)

            self.a_in = InPort(shape=shape)
            self.s_out = OutPort(shape=shape)

            self.alpha = Var(shape=(n,), init=alpha)
            self.tau_hab = Var(shape=(n,), init=tau_hab)
            self.tau_rec = Var(shape=(n,), init=tau_rec)
            self.novelty_threshold = Var(shape=(1,), init=novelty_threshold)
            self.familiarity_trace = Var(shape=(n,), init=0.0)
            self.history = Var(shape=(history_length, n), init=0.0)
            self.history_idx = Var(shape=(1,), init=0)
            self.history_length = Var(shape=(1,), init=history_length)

    @implements(proc=HabituationProcess, protocol=...)
    @requires(CPU)
    class PyHabituationModel(PyLoihiProcessModel):
        """Python implementation of HabituationProcess for CPU execution."""

        a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
        s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

        alpha: np.ndarray = LavaPyType(np.ndarray, float)
        tau_hab: np.ndarray = LavaPyType(np.ndarray, float)
        tau_rec: np.ndarray = LavaPyType(np.ndarray, float)
        novelty_threshold: np.ndarray = LavaPyType(np.ndarray, float)
        familiarity_trace: np.ndarray = LavaPyType(np.ndarray, float)
        history: np.ndarray = LavaPyType(np.ndarray, float)
        history_idx: np.ndarray = LavaPyType(np.ndarray, int)
        history_length: np.ndarray = LavaPyType(np.ndarray, int)

        def run_spk(self):
            current = self.a_in.recv()
            flat = current.flatten()

            # Compute novelty
            mean_hist = self.history.mean(axis=0)
            diff = np.abs(flat - mean_hist)
            max_val = np.maximum(np.abs(flat), np.abs(mean_hist))
            max_val = np.clip(max_val, 1e-8, None)
            novelty = diff / max_val

            # Update familiarity
            is_familiar = (novelty < self.novelty_threshold[0]).astype(float)
            self.familiarity_trace = np.clip(
                is_familiar * (self.tau_hab * self.familiarity_trace + (1 - self.tau_hab))
                + (1 - is_familiar) * (self.tau_rec * self.familiarity_trace),
                0.0, 1.0,
            )

            # Modulate
            gain = 1.0 - self.alpha * self.familiarity_trace
            output = flat * gain

            # Update history (circular buffer)
            idx = int(self.history_idx[0])
            self.history[idx] = flat
            self.history_idx[0] = (idx + 1) % int(self.history_length[0])

            self.s_out.send(output.reshape(current.shape))


def load_hamt_params_to_lava(pytorch_hab_layer, lava_process):
    """
    Transfer learned HAMT parameters from PyTorch to a Lava Process.

    After training with PyTorch + snnTorch, call this to copy
    the learned habituation parameters to the Lava deployment.

    Args:
        pytorch_hab_layer: Trained hamt.HabituationLayer instance
        lava_process: HabituationProcess instance
    """
    check_lava()
    import torch

    with torch.no_grad():
        lava_process.alpha.set(pytorch_hab_layer.alpha.cpu().numpy())
        lava_process.tau_hab.set(pytorch_hab_layer.tau_hab.cpu().numpy())
        lava_process.tau_rec.set(pytorch_hab_layer.tau_rec.cpu().numpy())


def pytorch_to_lava_deployment_guide():
    """Print step-by-step guide for deploying HAMT on Loihi via Lava."""
    return """
HAMT Deployment on Intel Loihi via Lava
========================================

Step 1: Train with PyTorch + snnTorch (as normal)
    from hamt import habituate_snn, MetabolicLoss
    model = habituate_snn(your_model)
    # ... train ...

Step 2: Extract learned parameters
    hab_layers = [m for m in model.modules() if isinstance(m, HabituationLayer)]

Step 3: Build Lava network
    from lava.proc.dense.process import Dense
    from lava.proc.lif.process import LIF
    from hamt.lava_compat import HabituationProcess, load_hamt_params_to_lava

    # For each layer: Dense -> Habituation -> LIF
    dense1 = Dense(weights=model.fc1.weight.detach().numpy())
    hab1 = HabituationProcess(shape=(800,))
    lif1 = LIF(shape=(800,), du=..., dv=..., vth=...)

    dense1.s_out.connect(hab1.a_in)
    hab1.s_out.connect(lif1.a_in)

Step 4: Load trained habituation parameters
    load_hamt_params_to_lava(hab_layers[0], hab1)

Step 5: Run on Loihi
    from lava.magma.core.run_configs import Loihi2HwCfg
    network.run(condition=..., run_cfg=Loihi2HwCfg())
"""
