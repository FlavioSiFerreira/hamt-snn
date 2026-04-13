"""
EMNIST Letters with DEFAULT config on GPU.

Tests whether default lambdas (0.001/0.0005) with h=800 show energy
savings on EMNIST, which showed 0% with aggressive lambdas on Kaggle.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.models.baseline_snn import BaselineSNN
from src.models.hamt_snn import HAMTSNN
from src.training.trainer import train_baseline, train_hamt
from src.utils.metrics import estimate_energy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
tr = datasets.EMNIST("data", split="letters", train=True, download=True, transform=tf)
te = datasets.EMNIST("data", split="letters", train=False, download=True, transform=tf)
trl = DataLoader(tr, 128, shuffle=True, drop_last=True)
tel = DataLoader(te, 128, shuffle=False)
print(f"EMNIST Letters: {len(tr)} train, {len(te)} test", flush=True)

print("\n=== BASELINE h=800 (15 epochs) ===", flush=True)
torch.manual_seed(42)
np.random.seed(42)
bh = train_baseline(
    BaselineSNN(input_size=784, hidden_size=800, output_size=27),
    trl, tel, num_epochs=15, device=device,
)

print("\n=== HAMT h=800 DEFAULT LAMBDAS (15 epochs) ===", flush=True)
torch.manual_seed(42)
np.random.seed(42)
hh = train_hamt(
    HAMTSNN(input_size=784, hidden_size=800, output_size=27),
    trl, tel, num_epochs=15,
    lambda_energy=0.001, lambda_habituation=0.0005,
    target_spike_rate=0.05, ramp_epochs=10, device=device,
)

ba, ha = max(bh["test_acc"]), max(hh["test_acc"])
bsr, hsr = bh["spike_rate"][-1], hh["spike_rate"][-1]
be, he = bh["energy_joules"][-1], hh["energy_joules"][-1]

print(f"\n{'='*60}", flush=True)
print(f"EMNIST LETTERS DEFAULT LAMBDAS RESULT", flush=True)
print(f"Accuracy: B={ba:.4f} H={ha:.4f} (delta={ha-ba:+.4f})", flush=True)
print(f"Spike Rate: B={bsr:.4f} H={hsr:.4f} (reduction={((bsr-hsr)/bsr)*100:.1f}%)", flush=True)
print(f"Energy: B={be:.2e} H={he:.2e} (reduction={((be-he)/be)*100:.1f}%)", flush=True)
