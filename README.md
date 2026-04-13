# HAMT: Habituation-Aware Metabolic Training

Drop-in habituation and metabolic-cost modules for spiking neural networks built on [snnTorch](https://github.com/jeshraghian/snntorch).

HAMT adds a within-pass suppression term inspired by biological repetition suppression. The habituation layer suppresses temporally redundant spikes inside a single forward pass (across the 25 simulation timesteps), which empirically reduces estimated inference energy on pattern recognition tasks with little or no accuracy loss.

No cross-sample adaptation is claimed. Attempts to add persistent cross-sample memory (EMA-based) were measured and did not improve over the within-pass version. The effect also does not compound with longer training. In practice the module behaves as an energy-cost regularizer, not as a true persistent habituation mechanism.

This repository contains the full code used to evaluate the approach. It is released as-is for anyone wanting to reuse the habituation layer, the metabolic loss, or the benchmark scripts.

---

## What is in this repo

```
hamt/                  Installable package (HabituationLayer, MetabolicLoss, wrappers)
src/                   Alternative reference implementation (habituation_module, metabolic_loss, models)
experiments/           Local GPU training scripts (ablation, MNIST, Fashion-MNIST, EMNIST, neuromorphic)
kaggle/                Kaggle T4 variants (long training, SHD, Fashion-MNIST, activity decay, synaptic depression)
demo/                  15-minute Fashion-MNIST proof script
BENCHMARK_REPORT.txt   Consolidated results across all runs
pyproject.toml         Package metadata
requirements.txt       Runtime dependencies
LICENSE                MIT
```

---

## Installation

Python 3.10 to 3.12, PyTorch 2.x. Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package (editable, from the repo root):

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[datasets]"  # adds tonic for neuromorphic datasets
pip install -e ".[viz]"       # adds matplotlib for plotting
pip install -e ".[full]"      # all of the above plus pandas
```

---

## Quick start

```python
import torch
import torch.nn as nn
import snntorch as snn
from hamt import habituate_snn, MetabolicLoss

# A standard snnTorch model.
model = nn.Sequential(
    nn.Linear(784, 800),
    snn.Leaky(beta=0.95),
    nn.Linear(800, 10),
    snn.Leaky(beta=0.95, output=True),
)

# Wrap it with HAMT (one function call).
hamt_model = habituate_snn(model, num_steps=25)

# Metabolic loss combines task loss with spike-rate and suppression terms.
loss_fn = MetabolicLoss(lambda_energy=0.001, lambda_habituation=0.0005)
```

A runnable end-to-end example is in `demo/hamt_demo.py` (Fashion-MNIST baseline vs HAMT, default hyperparameters, 15 epochs, runs on a free Colab T4).

---

## What was tested

All runs used default hyperparameters (`lambda_energy=0.001`, `lambda_habituation=0.0005`, `target_spike_rate=0.05`, `ramp_epochs=10`, `num_steps=25`, hidden size 800 for feedforward or Conv16-Conv32-FC for convolutional, seed 42). Energy estimated with Loihi 2 unit costs (120 pJ per spike, 23 pJ per synaptic operation).

**Hardware**

* Local: NVIDIA GTX 1650 (CUDA 11.8), Windows 11, Python 3.12
* Kaggle: T4 GPU, Python 3.10
* Google Colab: T4 free tier

**Datasets**

* MNIST, Fashion-MNIST, EMNIST (Letters, Balanced, Digits): pattern recognition, effect present
* CIFAR-10, SVHN: natural images, near-zero or negative effect
* SHD on Kaggle via `tonic`: neuromorphic event audio, modest 4.2% energy reduction at 15 epochs
* N-MNIST local via `tonic`: baseline already very sparse (~3% spike rate), minimal headroom
* DVS-Gesture: not tested (IBM license requires manual download)

---

## Observed results (from BENCHMARK_REPORT.txt)

Reported reductions are estimated inference energy. Accuracy delta is HAMT minus baseline.

| Dataset | Arch | Acc Δ | Energy reduction |
|---|---|---|---|
| MNIST (FF 800) | feedforward | -0.09% | 26.2% |
| MNIST (Conv) | convolutional | -0.05% to +0.10% | 15.8% to 19.2% |
| Fashion-MNIST (FF 800, local) | feedforward | -0.10% | 14.9% |
| Fashion-MNIST (FF 800, Kaggle T4) | feedforward | +0.41% | 40.7% |
| EMNIST Letters (FF 800) | feedforward | +0.52% to +0.91% | 9.2% to 16.8% |
| EMNIST Digits (FF 800) | feedforward | +0.04% | 16.9% |
| CIFAR-10 (Conv) | convolutional | -0.59% | 1.3% |
| SHD (Kaggle, 15 ep) | feedforward | +1.37% | 4.2% |

**100-epoch long training on Fashion-MNIST (T4):** energy reduction is stable around 36 to 44% across the whole training run and does not compound. HAMT accuracy at 100 epochs was +1.25% over baseline, where baseline overfits.

**Ablation on MNIST (15 epochs):** the habituation term is responsible for essentially all of the effect. Habituation alone gives ~25% reduction with the best accuracy (98.17%). The metabolic-penalty term alone gives ~0.5% reduction (useless). Combining both gives 22.3% reduction (slightly worse than habituation alone).

---

## Honest advantages and limitations

**Advantages**

* One-line integration with snnTorch models via `habituate_snn`
* Reduces estimated inference energy by roughly 20 to 40% on pattern recognition tasks without losing accuracy, sometimes with a small accuracy gain
* Slight regularization effect visible at longer training (baseline overfits, HAMT does not, on Fashion-MNIST 100 epochs)
* Pure PyTorch, no custom CUDA, runs on any snnTorch-compatible hardware
* Ablation-proven: the habituation term is the active ingredient

**Limitations**

* Reduction does not compound with longer training. Both baseline and HAMT spike rates rise in parallel, so the gap stays proportional rather than growing.
* Effect disappears on natural image classification (CIFAR-10, SVHN).
* Neuromorphic event-driven datasets show modest gains at 15 epochs (SHD 4%, N-MNIST near zero).
* Mechanism acts within a single forward pass only (`num_steps` timesteps). It does not accumulate across training samples. Attempts to add persistent cross-sample memory (EMA-based) performed worse than the original.
* An alternative path (activity-based weight decay after the gradient step) achieves ~68% reduction over 50 epochs on MNIST, but this is effectively learned pruning rather than a biological habituation mechanism.

If you are looking for true persistent habituation across training samples, HAMT in its current form does not deliver that. It is a solid efficient-inference regularizer for pattern-recognition SNNs.

---

## Reproducing results

```bash
# Ablation (4 conditions, MNIST, 15 epochs, around 10 min on GTX 1650)
python experiments/run_ablation.py

# Baseline vs HAMT on MNIST
python experiments/run_mnist_comparison.py

# 50-epoch activity-decay sweep
python experiments/run_activity_decay_full.py

# Neuromorphic (requires tonic extra)
python experiments/run_neuromorphic_local.py
```

Kaggle variants under `kaggle/` are structured for T4 runtime. The file named `hamt_benchmark_v1_AGGRESSIVE_DO_NOT_USE.py` is kept as a record only; it uses aggressive lambdas (0.01 and 0.005 with hidden size 400) that empirically give worse results than defaults.

---

## Citation

If you use this code, please cite the repository:

```
@software{ferreira_hamt_2026,
  author = {Ferreira, Flavio Da Silva},
  title  = {HAMT: Habituation-Aware Metabolic Training for Spiking Neural Networks},
  year   = {2026},
  url    = {https://github.com/FlavioSiFerreira/hamt-snn}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
