"""
HAMT Long-Training Experiment: Does Habituation Compound Over Time?

Hypothesis: 15 epochs barely starts habituation. With 100 epochs,
the habituation parameters should learn increasingly refined
suppression patterns, and energy savings should grow over time.

This is the "compounding returns" experiment. If energy reduction
grows from 15% at epoch 15 to 40%+ at epoch 100, the business
pitch becomes: "savings compound the longer you train."

Dataset: Fashion-MNIST (strongest HAMT result at 15 epochs: 40.7%)
Config: Default HAMT, h=800, 100 epochs
Output: JSON with per-epoch energy for both baseline and HAMT

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

# Config
LAMBDA_ENERGY = 0.001
LAMBDA_HAB = 0.0005
TARGET_SR = 0.05
RAMP_EPOCHS = 10
NUM_STEPS = 25
HIDDEN = 800
NUM_EPOCHS = 100  # THE KEY CHANGE: 100 instead of 15
BATCH_SIZE = 128
LR = 5e-4
SEED = 42

ENERGY_PER_SPIKE_PJ = 120.0
ENERGY_PER_SYNOP_PJ = 23.0

# ============================================================
# CELL 3: HAMT Core (self-contained)
# ============================================================

def _inv_sigmoid(x):
    x = max(min(x, 0.999), 0.001)
    return -torch.log(torch.tensor(1.0 / x - 1.0)).item()

class HabituationState:
    __slots__ = ("familiarity_trace", "input_history", "habituation_strength")
    def __init__(self, familiarity_trace, input_history, habituation_strength):
        self.familiarity_trace = familiarity_trace
        self.input_history = input_history
        self.habituation_strength = habituation_strength
    @staticmethod
    def initialize(batch_size, num_neurons, history_length=5, device=torch.device("cpu")):
        return HabituationState(
            torch.zeros(batch_size, num_neurons, device=device),
            torch.zeros(history_length, batch_size, num_neurons, device=device),
            torch.zeros(batch_size, num_neurons, device=device),
        )

class HabituationLayer(nn.Module):
    def __init__(self, num_neurons, alpha=0.5, tau_hab=0.9, tau_rec=0.8,
                 novelty_threshold=0.3, history_length=5):
        super().__init__()
        self.num_neurons = num_neurons
        self.history_length = history_length
        self.novelty_threshold = novelty_threshold
        self._alpha_raw = nn.Parameter(torch.full((num_neurons,), _inv_sigmoid(alpha)))
        self._tau_hab_raw = nn.Parameter(torch.full((num_neurons,), _inv_sigmoid(tau_hab)))
        self._tau_rec_raw = nn.Parameter(torch.full((num_neurons,), _inv_sigmoid(tau_rec)))
    @property
    def alpha(self): return torch.sigmoid(self._alpha_raw)
    @property
    def tau_hab(self): return torch.sigmoid(self._tau_hab_raw)
    @property
    def tau_rec(self): return torch.sigmoid(self._tau_rec_raw)

    def forward(self, current_input, state):
        mean_history = state.input_history.mean(dim=0)
        diff = torch.abs(current_input - mean_history)
        max_val = torch.max(torch.abs(current_input), torch.abs(mean_history)).clamp(min=1e-8)
        novelty = diff / max_val
        is_familiar = (novelty < self.novelty_threshold).float()
        new_familiarity = (
            is_familiar * (self.tau_hab * state.familiarity_trace + (1 - self.tau_hab))
            + (1 - is_familiar) * (self.tau_rec * state.familiarity_trace)
        ).clamp(0.0, 1.0)
        new_strength = self.alpha * new_familiarity
        modulated = current_input * (1.0 - new_strength)
        new_history = torch.cat([current_input.unsqueeze(0), state.input_history[:-1]], dim=0)
        return modulated, HabituationState(new_familiarity, new_history, new_strength)

class MetabolicLoss(nn.Module):
    def __init__(self, lambda_energy=0.001, lambda_habituation=0.0005,
                 target_spike_rate=0.05, ramp_epochs=10):
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_habituation = lambda_habituation
        self.target_spike_rate = target_spike_rate
        self.ramp_epochs = ramp_epochs
        self.task_loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, spike_recs, hab_strengths=None, epoch=0):
        l_task = self.task_loss_fn(predictions, targets)
        ramp = min(1.0, epoch / max(self.ramp_epochs, 1))
        total_spk = sum(s.sum() for s in spike_recs)
        total_n = sum(s.numel() for s in spike_recs)
        sr = total_spk / max(total_n, 1)
        excess = torch.relu(sr - self.target_spike_rate)
        l_energy = (excess ** 2 + 0.1 * sr) * ramp
        l_hab = torch.tensor(0.0, device=predictions.device)
        if hab_strengths is not None:
            for spk, hs in zip(spike_recs, hab_strengths):
                l_hab = l_hab + (hs * spk.mean(dim=0)).mean()
            l_hab = l_hab * ramp
        total = l_task + self.lambda_energy * l_energy + self.lambda_habituation * l_hab
        return total, {"task": l_task.item(), "energy": l_energy.item(),
                       "hab": l_hab.item(), "total": total.item()}

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
        return s3r.sum(0), [s1r, s2r, s3r], None

class HAMTSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.hab1 = HabituationLayer(HIDDEN)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.hab2 = HabituationLayer(HIDDEN)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
        self.fan_outs = [HIDDEN, HIDDEN, 10]
    def forward(self, x):
        bs = x.shape[1]
        mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        hs1 = HabituationState.initialize(bs, HIDDEN, device=x.device)
        hs2 = HabituationState.initialize(bs, HIDDEN, device=x.device)
        s1r, s2r, s3r = [], [], []
        for t in range(x.shape[0]):
            c1 = self.fc1(x[t])
            c1, hs1 = self.hab1(c1, hs1)
            spk1, mem1 = self.lif1(c1, mem1)
            c2 = self.fc2(spk1)
            c2, hs2 = self.hab2(c2, hs2)
            spk2, mem2 = self.lif2(c2, mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            s1r.append(spk1); s2r.append(spk2); s3r.append(spk3)
        s1r, s2r, s3r = torch.stack(s1r), torch.stack(s2r), torch.stack(s3r)
        return s3r.sum(0), [s1r, s2r, s3r], [hs1.habituation_strength, hs2.habituation_strength]

# ============================================================
# CELL 5: Data
# ============================================================

def rate_encode(images, num_steps):
    flat = images.view(images.shape[0], -1).clamp(0.0, 1.0)
    return (torch.rand(num_steps, flat.shape[0], flat.shape[1], device=flat.device) < flat.unsqueeze(0)).float()

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
train_ds = datasets.FashionMNIST("/kaggle/working/data", train=True, download=True, transform=tf)
test_ds = datasets.FashionMNIST("/kaggle/working/data", train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)
print(f"Fashion-MNIST: {len(train_ds)} train, {len(test_ds)} test", flush=True)

# ============================================================
# CELL 6: Training with per-epoch tracking
# ============================================================

def estimate_energy(spike_recs, fan_outs):
    total_spk = sum(int(s.sum().item()) for s in spike_recs)
    syn_ops = sum(int(s.sum().item()) * fo for s, fo in zip(spike_recs, fan_outs))
    return (total_spk * ENERGY_PER_SPIKE_PJ + syn_ops * ENERGY_PER_SYNOP_PJ) * 1e-12

def train_long(model, is_hamt, label):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if is_hamt:
        loss_fn = MetabolicLoss(
            lambda_energy=LAMBDA_ENERGY, lambda_habituation=LAMBDA_HAB,
            target_spike_rate=TARGET_SR, ramp_epochs=RAMP_EPOCHS,
        )
    else:
        loss_fn = nn.CrossEntropyLoss()

    history = {
        "test_acc": [], "spike_rate": [], "energy": [],
        "epoch_time": [], "train_acc": [],
    }

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        correct = total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            spk_in = rate_encode(imgs, NUM_STEPS)
            out, spike_recs, hab = model(spk_in)

            if is_hamt:
                loss, _ = loss_fn(out, labels, spike_recs, hab, epoch)
            else:
                loss = loss_fn(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)

        # Test eval
        model.eval()
        test_correct = test_total = 0
        all_sr = []
        all_energy = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                spk_in = rate_encode(imgs, NUM_STEPS)
                out, spike_recs, hab = model(spk_in)
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
        history["train_acc"].append(train_acc)
        history["spike_rate"].append(avg_sr)
        history["energy"].append(avg_energy)
        history["epoch_time"].append(epoch_time)

        # Print every 5 epochs to reduce noise
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{label} Epoch {epoch+1}/{NUM_EPOCHS}: "
                  f"TestAcc={test_acc:.4f} SR={avg_sr:.4f} "
                  f"E={avg_energy:.2e} T={epoch_time:.0f}s", flush=True)

        # Checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            with open(RESULTS_DIR / f"{label.lower()}_checkpoint_e{epoch+1}.json", "w") as f:
                json.dump(history, f, indent=2)
            print(f"  Checkpoint saved at epoch {epoch+1}", flush=True)

    return history

# ============================================================
# CELL 7: Run both models for 100 epochs
# ============================================================

print("=" * 70, flush=True)
print("HAMT LONG-TRAINING EXPERIMENT: Fashion-MNIST, 100 Epochs", flush=True)
print(f"Config: h={HIDDEN}, T={NUM_STEPS}, lr={LR}", flush=True)
print(f"Lambdas: energy={LAMBDA_ENERGY}, hab={LAMBDA_HAB}", flush=True)
print(f"Device: {DEVICE}", flush=True)
print("Hypothesis: habituation compounds over time", flush=True)
print("=" * 70, flush=True)

# Baseline 100 epochs
print(f"\n--- Baseline FF h={HIDDEN}, 100 epochs ---", flush=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
bh = train_long(BaselineSNN(), is_hamt=False, label="Base")
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# HAMT 100 epochs
print(f"\n--- HAMT FF h={HIDDEN}, 100 epochs ---", flush=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
hh = train_long(HAMTSNN(), is_hamt=True, label="HAMT")
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# CELL 8: Analysis and results
# ============================================================

print(f"\n{'='*70}", flush=True)
print("LONG-TRAINING RESULTS: Fashion-MNIST", flush=True)
print(f"{'='*70}", flush=True)

# Energy reduction at different epoch milestones
milestones = [10, 15, 25, 50, 75, 100]
print(f"\n{'Epoch':>8} {'B Acc':>8} {'H Acc':>8} {'B Energy':>12} "
      f"{'H Energy':>12} {'E Red%':>8} {'SR Red%':>8}", flush=True)
print("-" * 75, flush=True)

results_by_epoch = {}
for ep in milestones:
    idx = ep - 1
    if idx < len(bh["energy"]):
        b_e = bh["energy"][idx]
        h_e = hh["energy"][idx]
        e_red = ((b_e - h_e) / max(b_e, 1e-10)) * 100
        b_sr = bh["spike_rate"][idx]
        h_sr = hh["spike_rate"][idx]
        sr_red = ((b_sr - h_sr) / max(b_sr, 1e-10)) * 100

        print(f"{ep:>8} {bh['test_acc'][idx]:>8.4f} {hh['test_acc'][idx]:>8.4f} "
              f"{b_e:>12.2e} {h_e:>12.2e} {e_red:>+8.1f} {sr_red:>+8.1f}", flush=True)

        results_by_epoch[ep] = {
            "baseline_acc": bh["test_acc"][idx],
            "hamt_acc": hh["test_acc"][idx],
            "baseline_energy": b_e,
            "hamt_energy": h_e,
            "energy_reduction_pct": round(e_red, 1),
            "baseline_sr": b_sr,
            "hamt_sr": h_sr,
            "sr_reduction_pct": round(sr_red, 1),
        }

# Best results
b_best_acc = max(bh["test_acc"])
h_best_acc = max(hh["test_acc"])
b_final_e = bh["energy"][-1]
h_final_e = hh["energy"][-1]
final_e_red = ((b_final_e - h_final_e) / max(b_final_e, 1e-10)) * 100

print(f"\nBest Accuracy: Baseline={b_best_acc:.4f} HAMT={h_best_acc:.4f}")
print(f"Final Energy Reduction (epoch 100): {final_e_red:.1f}%")
print(f"Energy Reduction at epoch 15: {results_by_epoch.get(15, {}).get('energy_reduction_pct', 'N/A')}%")

if 15 in results_by_epoch and 100 in results_by_epoch:
    e15 = results_by_epoch[15]["energy_reduction_pct"]
    e100 = results_by_epoch[100]["energy_reduction_pct"]
    print(f"\nCOMPOUNDING FACTOR: {e100/max(e15, 0.1):.1f}x improvement from epoch 15 to 100")

# Save full results
full_results = {
    "experiment": "long_training_fashion_mnist",
    "num_epochs": NUM_EPOCHS,
    "config": {
        "hidden": HIDDEN, "steps": NUM_STEPS, "lr": LR,
        "lambda_energy": LAMBDA_ENERGY, "lambda_hab": LAMBDA_HAB,
        "target_sr": TARGET_SR, "ramp_epochs": RAMP_EPOCHS,
    },
    "milestones": results_by_epoch,
    "baseline_history": bh,
    "hamt_history": hh,
    "total_time_sec": sum(bh["epoch_time"]) + sum(hh["epoch_time"]),
}

with open(RESULTS_DIR / "long_training_results.json", "w") as f:
    json.dump(full_results, f, indent=2, default=str)

print(f"\nTotal time: {full_results['total_time_sec']/3600:.1f} hours", flush=True)
print(f"Results saved to: {RESULTS_DIR / 'long_training_results.json'}", flush=True)
print("EXPERIMENT COMPLETE", flush=True)
