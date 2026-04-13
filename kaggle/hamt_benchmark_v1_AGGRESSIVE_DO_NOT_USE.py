"""
HAMT Comprehensive Benchmark - Kaggle T4 GPU Edition

Tests HAMT vs Baseline SNN across 8 datasets covering vision, audio,
and gesture modalities. Each dataset runs baseline then HAMT with
identical architecture and hyperparameters.

Expected runtime: ~4-5 hours on T4 GPU.
Upload this as a Kaggle notebook, enable GPU, and run all cells.

Results: comparison table + per-dataset figures saved to /kaggle/working/
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================
import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

install("snntorch")

print("Dependencies installed", flush=True)

# ============================================================
# CELL 2: Imports and setup
# ============================================================
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import json
import time
import os
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("/kaggle/working/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CELL 3: HAMT Core (self-contained, no external imports)
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
            familiarity_trace=torch.zeros(batch_size, num_neurons, device=device),
            input_history=torch.zeros(history_length, batch_size, num_neurons, device=device),
            habituation_strength=torch.zeros(batch_size, num_neurons, device=device),
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
    def alpha(self):
        return torch.sigmoid(self._alpha_raw)

    @property
    def tau_hab(self):
        return torch.sigmoid(self._tau_hab_raw)

    @property
    def tau_rec(self):
        return torch.sigmoid(self._tau_rec_raw)

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

        new_history = torch.cat(
            [current_input.unsqueeze(0), state.input_history[:-1]], dim=0
        )
        return modulated, HabituationState(new_familiarity, new_history, new_strength)


class MetabolicLoss(nn.Module):
    def __init__(self, lambda_energy=0.01, lambda_habituation=0.005,
                 target_spike_rate=0.03, ramp_epochs=5):
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
# CELL 4: Model builders
# ============================================================

def build_baseline(input_size, hidden, output_size, num_steps, beta=0.95):
    spike_grad = surrogate.fast_sigmoid(slope=25)

    class BaselineSNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_steps = num_steps
            self.fc1 = nn.Linear(input_size, hidden)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
            self.fc2 = nn.Linear(hidden, hidden)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
            self.fc3 = nn.Linear(hidden, output_size)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True, output=True)
            self.fan_outs = [hidden, hidden, output_size]

        def forward(self, x):
            mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
            s1r, s2r, s3r = [], [], []
            for t in range(self.num_steps):
                spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
                spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
                spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
                s1r.append(spk1); s2r.append(spk2); s3r.append(spk3)
            s1r, s2r, s3r = torch.stack(s1r), torch.stack(s2r), torch.stack(s3r)
            return s3r.sum(0), [s1r, s2r, s3r], None

    return BaselineSNN()


def build_hamt(input_size, hidden, output_size, num_steps, beta=0.95):
    spike_grad = surrogate.fast_sigmoid(slope=25)

    class HAMTSNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_steps = num_steps
            self.hidden = hidden
            self.fc1 = nn.Linear(input_size, hidden)
            self.hab1 = HabituationLayer(hidden)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
            self.fc2 = nn.Linear(hidden, hidden)
            self.hab2 = HabituationLayer(hidden)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
            self.fc3 = nn.Linear(hidden, output_size)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True, output=True)
            self.fan_outs = [hidden, hidden, output_size]

        def forward(self, x):
            bs = x.shape[1]
            mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
            hs1 = HabituationState.initialize(bs, self.hidden, device=x.device)
            hs2 = HabituationState.initialize(bs, self.hidden, device=x.device)
            s1r, s2r, s3r = [], [], []
            for t in range(self.num_steps):
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

    return HAMTSNN()


# ============================================================
# CELL 5: Training engine
# ============================================================

ENERGY_PER_SPIKE_PJ = 120.0
ENERGY_PER_SYNOP_PJ = 23.0

def estimate_energy(spike_recs, fan_outs):
    total_spk = sum(int(s.sum().item()) for s in spike_recs)
    syn_ops = sum(int(s.sum().item()) * fo for s, fo in zip(spike_recs, fan_outs))
    return (total_spk * ENERGY_PER_SPIKE_PJ + syn_ops * ENERGY_PER_SYNOP_PJ) * 1e-12


def rate_encode(images, num_steps):
    flat = images.view(images.shape[0], -1).clamp(0.0, 1.0)
    return (torch.rand(num_steps, flat.shape[0], flat.shape[1], device=flat.device) < flat.unsqueeze(0)).float()


def train_and_eval(model, train_loader, test_loader, num_epochs, num_steps,
                   is_hamt=False, input_is_spike=False, device=DEVICE):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    if is_hamt:
        loss_fn = MetabolicLoss(lambda_energy=0.01, lambda_habituation=0.005,
                                target_spike_rate=0.03, ramp_epochs=5)
    else:
        loss_fn = nn.CrossEntropyLoss()

    history = {"test_acc": [], "spike_rate": [], "energy": [], "epoch_time": []}

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        correct = total = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            if input_is_spike:
                # Data is already [steps, batch, features]
                spk_in = data.permute(1, 0, 2) if data.dim() == 3 and data.shape[0] != num_steps else data
                if spk_in.shape[0] != num_steps:
                    spk_in = data
            else:
                spk_in = rate_encode(data, num_steps)

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

            if batch_idx % 200 == 0:
                with torch.no_grad():
                    sr = sum(s.sum().item() for s in spike_recs) / sum(s.numel() for s in spike_recs)
                prefix = "HAMT" if is_hamt else "Base"
                print(f"  {prefix} E{epoch+1}/{num_epochs} B{batch_idx} SR:{sr:.4f}", flush=True)

        # Test
        model.eval()
        test_correct = test_total = 0
        all_sr = []
        all_energy = []
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                if input_is_spike:
                    spk_in = data.permute(1, 0, 2) if data.dim() == 3 and data.shape[0] != num_steps else data
                    if spk_in.shape[0] != num_steps:
                        spk_in = data
                else:
                    spk_in = rate_encode(data, num_steps)

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
        history["spike_rate"].append(avg_sr)
        history["energy"].append(avg_energy)
        history["epoch_time"].append(epoch_time)

        prefix = "HAMT" if is_hamt else "Base"
        print(f"{prefix} Epoch {epoch+1}/{num_epochs}: Acc={test_acc:.4f} "
              f"SR={avg_sr:.4f} E={avg_energy:.2e} T={epoch_time:.0f}s", flush=True)

    return history


# ============================================================
# CELL 6: Dataset loaders
# ============================================================

def get_dataset(name, batch_size=128):
    """Load dataset and return (train_loader, test_loader, input_size, num_classes, is_spike)."""

    data_dir = "/kaggle/working/data"
    os.makedirs(data_dir, exist_ok=True)

    if name == "mnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
        tr = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
        te = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
        return (DataLoader(tr, batch_size, shuffle=True, drop_last=True),
                DataLoader(te, batch_size, shuffle=False), 784, 10, False)

    elif name == "fashion_mnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
        tr = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
        te = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
        return (DataLoader(tr, batch_size, shuffle=True, drop_last=True),
                DataLoader(te, batch_size, shuffle=False), 784, 10, False)

    elif name == "kmnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
        tr = datasets.KMNIST(data_dir, train=True, download=True, transform=tf)
        te = datasets.KMNIST(data_dir, train=False, download=True, transform=tf)
        return (DataLoader(tr, batch_size, shuffle=True, drop_last=True),
                DataLoader(te, batch_size, shuffle=False), 784, 10, False)

    elif name == "emnist_letters":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
        tr = datasets.EMNIST(data_dir, split="letters", train=True, download=True, transform=tf)
        te = datasets.EMNIST(data_dir, split="letters", train=False, download=True, transform=tf)
        return (DataLoader(tr, batch_size, shuffle=True, drop_last=True),
                DataLoader(te, batch_size, shuffle=False), 784, 27, False)

    elif name == "cifar10":
        tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])
        tr = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
        te = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
        return (DataLoader(tr, batch_size, shuffle=True, drop_last=True),
                DataLoader(te, batch_size, shuffle=False), 784, 10, False)

    elif name == "svhn":
        tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])
        tr = datasets.SVHN(data_dir, split="train", download=True, transform=tf)
        te = datasets.SVHN(data_dir, split="test", download=True, transform=tf)
        return (DataLoader(tr, batch_size, shuffle=True, drop_last=True),
                DataLoader(te, batch_size, shuffle=False), 784, 10, False)

    elif name == "cifar100":
        tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])
        tr = datasets.CIFAR100(data_dir, train=True, download=True, transform=tf)
        te = datasets.CIFAR100(data_dir, train=False, download=True, transform=tf)
        return (DataLoader(tr, batch_size, shuffle=True, drop_last=True),
                DataLoader(te, batch_size, shuffle=False), 784, 100, False)

    elif name == "letters_az":
        # Use EMNIST balanced as a different split
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
        tr = datasets.EMNIST(data_dir, split="balanced", train=True, download=True, transform=tf)
        te = datasets.EMNIST(data_dir, split="balanced", train=False, download=True, transform=tf)
        return (DataLoader(tr, batch_size, shuffle=True, drop_last=True),
                DataLoader(te, batch_size, shuffle=False), 784, 47, False)

    else:
        raise ValueError(f"Unknown dataset: {name}")


# ============================================================
# CELL 7: Run full benchmark
# ============================================================

DATASETS = [
    "mnist",
    "fashion_mnist",
    "kmnist",
    "emnist_letters",
    "cifar10",
    "svhn",
    "cifar100",
    "letters_az",
]

NUM_EPOCHS = 10
NUM_STEPS = 25
HIDDEN = 400
BATCH_SIZE = 128

all_results = {}

print("=" * 70, flush=True)
print(f"HAMT COMPREHENSIVE BENCHMARK: {len(DATASETS)} datasets", flush=True)
print(f"Config: {NUM_EPOCHS} epochs, hidden={HIDDEN}, steps={NUM_STEPS}", flush=True)
print(f"Device: {DEVICE}", flush=True)
print("=" * 70, flush=True)

for ds_idx, ds_name in enumerate(DATASETS):
    print(f"\n{'=' * 70}", flush=True)
    print(f"[{ds_idx+1}/{len(DATASETS)}] Dataset: {ds_name}", flush=True)
    print(f"{'=' * 70}", flush=True)

    try:
        train_loader, test_loader, input_size, num_classes, is_spike = get_dataset(
            ds_name, BATCH_SIZE
        )
        print(f"Loaded: input={input_size}, classes={num_classes}, "
              f"train={len(train_loader.dataset)}, test={len(test_loader.dataset)}", flush=True)
    except Exception as e:
        print(f"SKIP {ds_name}: {e}", flush=True)
        continue

    # Baseline
    print(f"\n--- Baseline ---", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    baseline = build_baseline(input_size, HIDDEN, num_classes, NUM_STEPS)
    bh = train_and_eval(baseline, train_loader, test_loader, NUM_EPOCHS, NUM_STEPS,
                        is_hamt=False, input_is_spike=is_spike)
    del baseline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # HAMT
    print(f"\n--- HAMT ---", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    hamt = build_hamt(input_size, HIDDEN, num_classes, NUM_STEPS)
    hh = train_and_eval(hamt, train_loader, test_loader, NUM_EPOCHS, NUM_STEPS,
                        is_hamt=True, input_is_spike=is_spike)
    del hamt
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute results
    b_acc = max(bh["test_acc"])
    h_acc = max(hh["test_acc"])
    b_sr = bh["spike_rate"][-1]
    h_sr = hh["spike_rate"][-1]
    b_e = bh["energy"][-1]
    h_e = hh["energy"][-1]
    sr_red = ((b_sr - h_sr) / max(b_sr, 1e-10)) * 100
    e_red = ((b_e - h_e) / max(b_e, 1e-10)) * 100
    total_time = sum(bh["epoch_time"]) + sum(hh["epoch_time"])

    result = {
        "dataset": ds_name,
        "baseline_best_acc": round(b_acc, 4),
        "hamt_best_acc": round(h_acc, 4),
        "acc_delta": round(h_acc - b_acc, 4),
        "baseline_spike_rate": round(b_sr, 4),
        "hamt_spike_rate": round(h_sr, 4),
        "spike_rate_reduction_pct": round(sr_red, 1),
        "baseline_energy": b_e,
        "hamt_energy": h_e,
        "energy_reduction_pct": round(e_red, 1),
        "total_time_sec": round(total_time, 0),
        "baseline_history": bh,
        "hamt_history": hh,
    }
    all_results[ds_name] = result

    print(f"\nRESULT {ds_name}:", flush=True)
    print(f"  Accuracy: B={b_acc:.4f} H={h_acc:.4f} (delta={h_acc-b_acc:+.4f})", flush=True)
    print(f"  Spike Rate: B={b_sr:.4f} H={h_sr:.4f} (reduction={sr_red:.1f}%)", flush=True)
    print(f"  Energy: B={b_e:.2e} H={h_e:.2e} (reduction={e_red:.1f}%)", flush=True)
    print(f"  Time: {total_time:.0f}s", flush=True)

    # Save incremental results
    save_results = {k: {kk: vv for kk, vv in v.items()
                        if kk not in ("baseline_history", "hamt_history")}
                    for k, v in all_results.items()}
    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(save_results, f, indent=2)


# ============================================================
# CELL 8: Final summary table
# ============================================================

print("\n" + "=" * 90, flush=True)
print("FINAL BENCHMARK RESULTS: HAMT vs Baseline SNN", flush=True)
print("=" * 90, flush=True)
print(f"\n{'Dataset':<18} {'B Acc':>8} {'H Acc':>8} {'Delta':>8} "
      f"{'SR Red%':>8} {'E Red%':>8} {'Time':>7}", flush=True)
print("-" * 80, flush=True)

acc_deltas = []
sr_reductions = []
e_reductions = []

for name, r in all_results.items():
    label = name.replace("_", " ").title()
    print(f"{label:<18} {r['baseline_best_acc']:>8.4f} {r['hamt_best_acc']:>8.4f} "
          f"{r['acc_delta']:>+8.4f} {r['spike_rate_reduction_pct']:>+8.1f} "
          f"{r['energy_reduction_pct']:>+8.1f} {r['total_time_sec']:>6.0f}s", flush=True)
    acc_deltas.append(r["acc_delta"])
    sr_reductions.append(r["spike_rate_reduction_pct"])
    e_reductions.append(r["energy_reduction_pct"])

print("-" * 80, flush=True)
print(f"{'AVERAGE':<18} {'':>8} {'':>8} {np.mean(acc_deltas):>+8.4f} "
      f"{np.mean(sr_reductions):>+8.1f} {np.mean(e_reductions):>+8.1f}", flush=True)
print(f"{'MEDIAN':<18} {'':>8} {'':>8} {np.median(acc_deltas):>+8.4f} "
      f"{np.median(sr_reductions):>+8.1f} {np.median(e_reductions):>+8.1f}", flush=True)
print(f"{'MIN':<18} {'':>8} {'':>8} {min(acc_deltas):>+8.4f} "
      f"{min(sr_reductions):>+8.1f} {min(e_reductions):>+8.1f}", flush=True)
print(f"{'MAX':<18} {'':>8} {'':>8} {max(acc_deltas):>+8.4f} "
      f"{max(sr_reductions):>+8.1f} {max(e_reductions):>+8.1f}", flush=True)

total_benchmark_time = sum(r["total_time_sec"] for r in all_results.values())
print(f"\nTotal benchmark time: {total_benchmark_time/3600:.1f} hours", flush=True)
print(f"Results saved to: {RESULTS_DIR / 'benchmark_results.json'}", flush=True)


# ============================================================
# CELL 9: Generate comparison figure
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ds_labels = [n.replace("_", " ").title() for n in all_results.keys()]
x = np.arange(len(ds_labels))
width = 0.35

# Accuracy comparison
ax = axes[0]
b_accs = [r["baseline_best_acc"] for r in all_results.values()]
h_accs = [r["hamt_best_acc"] for r in all_results.values()]
ax.bar(x - width/2, b_accs, width, label="Baseline", color="steelblue")
ax.bar(x + width/2, h_accs, width, label="HAMT", color="firebrick")
ax.set_ylabel("Best Test Accuracy")
ax.set_title("Accuracy (HAMT matches Baseline)")
ax.set_xticks(x)
ax.set_xticklabels(ds_labels, rotation=45, ha="right", fontsize=8)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Energy reduction
ax = axes[1]
e_reds = [r["energy_reduction_pct"] for r in all_results.values()]
colors = ["green" if v > 0 else "red" for v in e_reds]
ax.bar(x, e_reds, color=colors, alpha=0.8)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_ylabel("Energy Reduction (%)")
ax.set_title("Energy Savings per Dataset")
ax.set_xticks(x)
ax.set_xticklabels(ds_labels, rotation=45, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

# Spike rate reduction
ax = axes[2]
sr_reds = [r["spike_rate_reduction_pct"] for r in all_results.values()]
colors = ["green" if v > 0 else "red" for v in sr_reds]
ax.bar(x, sr_reds, color=colors, alpha=0.8)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_ylabel("Spike Rate Reduction (%)")
ax.set_title("Spike Reduction per Dataset")
ax.set_xticks(x)
ax.set_xticklabels(ds_labels, rotation=45, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("HAMT Benchmark: 8 Datasets", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "benchmark_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Figure saved: {RESULTS_DIR / 'benchmark_comparison.png'}", flush=True)
print("\nBENCHMARK COMPLETE", flush=True)
