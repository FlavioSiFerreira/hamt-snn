"""
Idea C (Activity-Proportional Weight Decay) on Fashion-MNIST, 100 epochs.

Rapid screen on MNIST showed 52% energy reduction in 10 epochs.
Fashion-MNIST has much higher baseline spike rates (climbs to 14%),
giving more room for compounding energy savings.

Mechanism: After each optimizer step, weights connected to neurons
firing above median rate get extra L2 decay. High-firing pathways
shrink over time. Standard CrossEntropyLoss only.

No tonic, no scipy. Kaggle-safe.
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "snntorch", "-q"])
print("snntorch installed", flush=True)

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import json
import time
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("/kaggle/working/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_STEPS = 25
HIDDEN = 800
NUM_EPOCHS = 100
BATCH_SIZE = 128
LR = 5e-4
SEED = 42

ENERGY_PER_SPIKE_PJ = 120.0
ENERGY_PER_SYNOP_PJ = 23.0

spike_grad = surrogate.fast_sigmoid(slope=25)


def estimate_energy(spike_recs, fan_outs):
    total_spk = sum(int(s.sum().item()) for s in spike_recs)
    syn_ops = sum(int(s.sum().item()) * fo for s, fo in zip(spike_recs, fan_outs))
    return (total_spk * ENERGY_PER_SPIKE_PJ + syn_ops * ENERGY_PER_SYNOP_PJ) * 1e-12


def rate_encode(images, num_steps):
    flat = images.view(images.shape[0], -1).clamp(0.0, 1.0)
    return (torch.rand(num_steps, flat.shape[0], flat.shape[1], device=flat.device) < flat.unsqueeze(0)).float()


class ActivityDecaySNN(nn.Module):
    def __init__(self, decay_rate=0.001):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
        self.fan_outs = [HIDDEN, HIDDEN, 10]
        self.decay_rate = decay_rate

        self.register_buffer('cum_fire1', torch.zeros(HIDDEN))
        self.register_buffer('cum_fire2', torch.zeros(HIDDEN))
        self.register_buffer('batch_count', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        s1r, s2r, s3r = [], [], []
        for t in range(x.shape[0]):
            spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            s1r.append(spk1); s2r.append(spk2); s3r.append(spk3)
        s1r, s2r, s3r = torch.stack(s1r), torch.stack(s2r), torch.stack(s3r)

        if self.training:
            with torch.no_grad():
                self.cum_fire1 += s1r.mean(dim=(0, 1))
                self.cum_fire2 += s2r.mean(dim=(0, 1))
                self.batch_count += 1

        return s3r.sum(0), [s1r, s2r, s3r]

    def apply_activity_decay(self):
        if self.batch_count == 0:
            return
        with torch.no_grad():
            avg_fire1 = self.cum_fire1 / self.batch_count
            avg_fire2 = self.cum_fire2 / self.batch_count
            mask1 = (avg_fire1 > avg_fire1.median()).float()
            mask2 = (avg_fire2 > avg_fire2.median()).float()
            self.fc1.weight.data *= 1 - self.decay_rate * mask1.unsqueeze(1).expand_as(self.fc1.weight)
            self.fc2.weight.data *= 1 - self.decay_rate * mask2.unsqueeze(1).expand_as(self.fc2.weight)


# Baseline (same as always)
class BaselineSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
        self.fan_outs = [HIDDEN, HIDDEN, 10]

    def forward(self, x):
        mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        s1r, s2r, s3r = [], [], []
        for t in range(x.shape[0]):
            spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            s1r.append(spk1); s2r.append(spk2); s3r.append(spk3)
        s1r, s2r, s3r = torch.stack(s1r), torch.stack(s2r), torch.stack(s3r)
        return s3r.sum(0), [s1r, s2r, s3r]


def train_model(model, train_loader, test_loader, label, use_decay=False):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = {"test_acc": [], "spike_rate": [], "energy": [], "epoch_time": []}

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out, spike_recs = model(rate_encode(imgs, NUM_STEPS))
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_decay:
                model.apply_activity_decay()

        model.eval()
        test_correct = test_total = 0
        all_sr = []
        all_energy = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out, spike_recs = model(rate_encode(imgs, NUM_STEPS))
                test_correct += (out.argmax(1) == labels).sum().item()
                test_total += labels.size(0)
                sr = sum(s.sum().item() for s in spike_recs) / sum(s.numel() for s in spike_recs)
                all_sr.append(sr)
                all_energy.append(estimate_energy(spike_recs, model.fan_outs))

        test_acc = test_correct / max(test_total, 1)
        avg_sr = np.mean(all_sr)
        avg_energy = np.mean(all_energy)
        epoch_time = time.time() - t0

        history["test_acc"].append(test_acc)
        history["spike_rate"].append(avg_sr)
        history["energy"].append(avg_energy)
        history["epoch_time"].append(epoch_time)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{label} Epoch {epoch+1}/{NUM_EPOCHS}: Acc={test_acc:.4f} "
                  f"SR={avg_sr:.4f} E={avg_energy:.2e}", flush=True)

        if (epoch + 1) % 25 == 0:
            with open(RESULTS_DIR / f"{label.lower()}_checkpoint_e{epoch+1}.json", "w") as f:
                json.dump(history, f, indent=2)
            print(f"  Checkpoint saved at epoch {epoch+1}", flush=True)

    return history


# Data
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
train_ds = datasets.FashionMNIST("/kaggle/working/data", train=True, download=True, transform=tf)
test_ds = datasets.FashionMNIST("/kaggle/working/data", train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)
print(f"Fashion-MNIST: {len(train_ds)} train, {len(test_ds)} test", flush=True)

# Run
print("=" * 70, flush=True)
print("IDEA C: ACTIVITY DECAY vs BASELINE, Fashion-MNIST, 100 Epochs", flush=True)
print(f"Config: h={HIDDEN}, T={NUM_STEPS}, decay_rate=0.001", flush=True)
print(f"Device: {DEVICE}", flush=True)
print("=" * 70, flush=True)

# Baseline
print(f"\n--- Baseline ---", flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)
bh = train_model(BaselineSNN(), train_loader, test_loader, "Base", use_decay=False)
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Activity Decay
print(f"\n--- Activity Decay ---", flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)
dh = train_model(ActivityDecaySNN(decay_rate=0.001), train_loader, test_loader, "Decay", use_decay=True)
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Results
print(f"\n{'='*70}", flush=True)
print("ACTIVITY DECAY vs BASELINE: Fashion-MNIST", flush=True)
print(f"{'='*70}", flush=True)

milestones = [5, 10, 15, 25, 50, 75, 100]
print(f"\n{'Epoch':>6} {'B Acc':>8} {'D Acc':>8} {'B Energy':>10} "
      f"{'D Energy':>10} {'Reduction':>10}", flush=True)
print("-" * 55, flush=True)

for ep in milestones:
    idx = ep - 1
    if idx < len(bh["energy"]):
        b_e = bh["energy"][idx]
        d_e = dh["energy"][idx]
        red = ((b_e - d_e) / max(b_e, 1e-10)) * 100
        print(f"{ep:>6} {bh['test_acc'][idx]:>8.4f} {dh['test_acc'][idx]:>8.4f} "
              f"{b_e:>10.2e} {d_e:>10.2e} {red:>+10.1f}%", flush=True)

# Compounding check
early_reds = []
late_reds = []
for ep in [5, 10, 15]:
    idx = ep - 1
    if idx < len(bh["energy"]):
        early_reds.append(((bh["energy"][idx] - dh["energy"][idx]) / bh["energy"][idx]) * 100)
for ep in [75, 100]:
    idx = ep - 1
    if idx < len(bh["energy"]):
        late_reds.append(((bh["energy"][idx] - dh["energy"][idx]) / bh["energy"][idx]) * 100)

if early_reds and late_reds:
    early = np.mean(early_reds)
    late = np.mean(late_reds)
    print(f"\nEarly avg (ep 5-15): {early:.1f}%", flush=True)
    print(f"Late avg (ep 75-100): {late:.1f}%", flush=True)
    print(f"Compounding ratio: {late/max(early, 0.1):.2f}x", flush=True)
    if late > early * 1.3:
        print("COMPOUNDING CONFIRMED", flush=True)

# Save
results = {"baseline": bh, "activity_decay": dh}
with open(RESULTS_DIR / "activity_decay_fashion_100ep.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved. EXPERIMENT COMPLETE", flush=True)
