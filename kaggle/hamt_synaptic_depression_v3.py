"""
Synaptic Use-Dependent Depression: Weight-Level Habituation

Instead of modulating input currents (current HAMT), this modulates
the WEIGHTS based on cumulative synapse usage across all training.

Biological basis: synaptic vesicle depletion. Frequently fired synapses
physically weaken. The brain doesn't suppress the input signal; it
weakens the connection itself when it's been overused.

This should compound over training because usage stats accumulate
across the entire dataset, not within a single sample.

Test: Fashion-MNIST, 50 epochs, baseline vs synaptic depression.
Track energy reduction at every epoch to see if the gap GROWS.

No tonic, no scipy. Kaggle-safe.
"""

# ============================================================
# CELL 1: Install
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "snntorch", "-q"])
print("snntorch installed", flush=True)

# ============================================================
# CELL 2: Imports
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
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
NUM_EPOCHS = 50
BATCH_SIZE = 128
LR = 5e-4
SEED = 42
TARGET_SR = 0.05

ENERGY_PER_SPIKE_PJ = 120.0
ENERGY_PER_SYNOP_PJ = 23.0

# ============================================================
# CELL 3: Synaptic Depression Module
# ============================================================

class DepressedLinear(nn.Module):
    """
    Linear layer with use-dependent synaptic depression.

    Wraps nn.Linear. Tracks cumulative activation per synapse
    (weight) using an EMA. Frequently used synapses get their
    effective weight reduced, forcing the network to find
    energy-efficient representations.

    The depression is NOT a learnable parameter. It's an emergent
    property of usage patterns, like biological vesicle depletion.
    """

    def __init__(self, in_features, out_features, bias=True,
                 usage_decay=0.999, depression_strength=5.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.usage_decay = usage_decay
        self.depression_strength = depression_strength

        # Persistent usage tracker (survives across batches)
        self.register_buffer(
            "synapse_usage",
            torch.zeros(out_features, in_features),
        )
        self.register_buffer(
            "batch_count",
            torch.tensor(0, dtype=torch.long),
        )

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        # Compute depression factor from cumulative usage
        # High usage -> depression approaches 1 -> effective weight approaches 0
        # Low usage -> depression approaches 0 -> full weight
        if self.batch_count > 0:
            # Normalize usage to [0, 1] range relative to max
            usage_norm = self.synapse_usage / (self.synapse_usage.max() + 1e-8)
            depression = torch.sigmoid(
                self.depression_strength * (usage_norm - 0.5)
            )
            # Scale depression: max ~30% weight reduction for most-used synapses
            depression = depression * 0.3
            effective_weight = self.linear.weight * (1.0 - depression)
        else:
            effective_weight = self.linear.weight

        # Forward pass with depressed weights
        out = F.linear(x, effective_weight, self.linear.bias)

        # Update usage statistics (only during training)
        if self.training:
            self._update_usage(x, out)

        return out

    @torch.no_grad()
    def _update_usage(self, x, out):
        """
        Update synapse usage with simplified activity measure.

        Uses absolute mean activation per neuron (not full outer product)
        to keep it memory-efficient. Approximation: if input neuron i
        and output neuron j are both active, synapse (j,i) is "used."
        """
        # Mean absolute activation per input/output neuron across batch
        in_activity = torch.abs(x).mean(dim=0)   # [in_features]
        out_activity = torch.abs(out).mean(dim=0)  # [out_features]

        # Outer product gives per-synapse activity estimate
        activity = out_activity.unsqueeze(1) * in_activity.unsqueeze(0)

        # EMA update
        d = self.usage_decay
        self.synapse_usage.mul_(d).add_(activity, alpha=1 - d)
        self.batch_count += 1

    def get_depression_stats(self):
        if self.batch_count == 0:
            return {"active": False}
        usage_norm = self.synapse_usage / (self.synapse_usage.max() + 1e-8)
        depression = torch.sigmoid(
            self.depression_strength * (usage_norm - 0.5)
        ) * 0.3
        return {
            "active": True,
            "batch_count": self.batch_count.item(),
            "mean_depression": depression.mean().item(),
            "max_depression": depression.max().item(),
            "mean_usage": self.synapse_usage.mean().item(),
            "pct_above_50": (usage_norm > 0.5).float().mean().item() * 100,
        }


# ============================================================
# CELL 4: Models
# ============================================================

spike_grad = surrogate.fast_sigmoid(slope=25)


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


class SynapticDepressionSNN(nn.Module):
    """
    SNN with use-dependent synaptic depression on all layers.
    No MetabolicLoss needed. The depression IS the energy mechanism.
    Trained with standard CrossEntropyLoss only.
    """

    def __init__(self, usage_decay=0.999, depression_strength=5.0):
        super().__init__()
        self.fc1 = DepressedLinear(784, HIDDEN,
                                    usage_decay=usage_decay,
                                    depression_strength=depression_strength)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc2 = DepressedLinear(HIDDEN, HIDDEN,
                                    usage_decay=usage_decay,
                                    depression_strength=depression_strength)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc3 = DepressedLinear(HIDDEN, 10,
                                    usage_decay=usage_decay,
                                    depression_strength=depression_strength)
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


# ============================================================
# CELL 5: Data
# ============================================================

def rate_encode(images, num_steps):
    flat = images.view(images.shape[0], -1).clamp(0.0, 1.0)
    return (torch.rand(num_steps, flat.shape[0], flat.shape[1], device=flat.device) < flat.unsqueeze(0)).float()


def estimate_energy(spike_recs, fan_outs):
    total_spk = sum(int(s.sum().item()) for s in spike_recs)
    syn_ops = sum(int(s.sum().item()) * fo for s, fo in zip(spike_recs, fan_outs))
    return (total_spk * ENERGY_PER_SPIKE_PJ + syn_ops * ENERGY_PER_SYNOP_PJ) * 1e-12


tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
train_ds = datasets.FashionMNIST("/kaggle/working/data", train=True, download=True, transform=tf)
test_ds = datasets.FashionMNIST("/kaggle/working/data", train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)
print(f"Fashion-MNIST: {len(train_ds)} train, {len(test_ds)} test", flush=True)


# ============================================================
# CELL 6: Training (both use standard CrossEntropyLoss)
# ============================================================

def train_model(model, label):
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
            dep_info = ""
            if hasattr(model, 'fc1') and hasattr(model.fc1, 'get_depression_stats'):
                stats = model.fc1.get_depression_stats()
                if stats["active"]:
                    dep_info = (f" dep={stats['mean_depression']:.3f} "
                                f"max={stats['max_depression']:.3f} "
                                f"usage50={stats['pct_above_50']:.0f}%")
            print(f"{label} Epoch {epoch+1}/{NUM_EPOCHS}: "
                  f"Acc={test_acc:.4f} SR={avg_sr:.4f} "
                  f"E={avg_energy:.2e}{dep_info}", flush=True)

    return history


# ============================================================
# CELL 7: Run experiment
# ============================================================

print("=" * 70, flush=True)
print("SYNAPTIC DEPRESSION EXPERIMENT: Fashion-MNIST, 50 Epochs", flush=True)
print(f"Config: h={HIDDEN}, T={NUM_STEPS}, lr={LR}", flush=True)
print("Both models use standard CrossEntropyLoss (no MetabolicLoss)", flush=True)
print("Depression is emergent from usage, not from a loss penalty", flush=True)
print("=" * 70, flush=True)

# Baseline
print(f"\n--- Baseline ---", flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)
bh = train_model(BaselineSNN(), "Base")
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Synaptic Depression
print(f"\n--- Synaptic Depression ---", flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)
sh = train_model(SynapticDepressionSNN(usage_decay=0.999, depression_strength=5.0), "SDep")
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================
# CELL 8: Results
# ============================================================

print(f"\n{'='*70}", flush=True)
print("SYNAPTIC DEPRESSION RESULTS", flush=True)
print(f"{'='*70}", flush=True)

milestones = [5, 10, 15, 25, 50]
print(f"\n{'Epoch':>6} {'Base Acc':>9} {'SDep Acc':>9} {'Base E':>10} "
      f"{'SDep E':>10} {'E Red%':>8}", flush=True)
print("-" * 60, flush=True)

for ep in milestones:
    idx = ep - 1
    if idx < len(bh["energy"]):
        b_e = bh["energy"][idx]
        s_e = sh["energy"][idx]
        e_red = ((b_e - s_e) / max(b_e, 1e-10)) * 100
        print(f"{ep:>6} {bh['test_acc'][idx]:>9.4f} {sh['test_acc'][idx]:>9.4f} "
              f"{b_e:>10.2e} {s_e:>10.2e} {e_red:>+8.1f}", flush=True)

# Check if reduction is GROWING (the key question)
reductions = []
for ep in [5, 10, 15, 25, 50]:
    idx = ep - 1
    if idx < len(bh["energy"]):
        b_e = bh["energy"][idx]
        s_e = sh["energy"][idx]
        reductions.append(((b_e - s_e) / max(b_e, 1e-10)) * 100)

if len(reductions) >= 3:
    early = np.mean(reductions[:2])
    late = np.mean(reductions[-2:])
    print(f"\nEarly avg reduction (ep 5-10): {early:.1f}%", flush=True)
    print(f"Late avg reduction (ep 25-50): {late:.1f}%", flush=True)
    if late > early * 1.2:
        print("COMPOUNDING CONFIRMED: late reduction > 120% of early", flush=True)
    else:
        print(f"Compounding ratio: {late/max(early,0.1):.2f}x (need >1.2x)", flush=True)

# Save
results = {
    "experiment": "synaptic_depression_fashion_mnist",
    "epochs": NUM_EPOCHS,
    "baseline": bh,
    "synaptic_depression": sh,
}
with open(RESULTS_DIR / "synaptic_depression_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved.", flush=True)
print("EXPERIMENT COMPLETE", flush=True)
