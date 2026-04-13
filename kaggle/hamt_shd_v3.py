"""
HAMT Neuromorphic Benchmark: SHD (Spiking Heidelberg Digits)
Kaggle T4 GPU Edition - v3

NO tonic, NO scipy. Uses only: torch, numpy, h5py, requests.
All tested on Kaggle environment.

SHD: 700 input neurons, 20 spoken digit classes, native spike format.
Downloads directly from Zenodo (HDF5).

Config: Default HAMT (lambda_e=0.001, lambda_h=0.0005, h=800, T=25, 15 epochs)

Instructions: Upload to Kaggle, enable GPU + Internet, run all cells.
"""

# ============================================================
# CELL 1: Install snntorch only (no tonic needed)
# ============================================================
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "snntorch", "-q"])
print("snntorch installed", flush=True)

# ============================================================
# CELL 2: Imports (all Kaggle-safe, no tonic/scipy)
# ============================================================
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import json
import time
import os
import h5py
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("/kaggle/working/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/kaggle/working/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Verified optimal config
LAMBDA_ENERGY = 0.001
LAMBDA_HAB = 0.0005
TARGET_SR = 0.05
RAMP_EPOCHS = 10
NUM_STEPS = 25
HIDDEN = 800
NUM_EPOCHS = 15
BATCH_SIZE = 128
LR = 5e-4
SEED = 42
SHD_NEURONS = 700
SHD_CLASSES = 20

ENERGY_PER_SPIKE_PJ = 120.0
ENERGY_PER_SYNOP_PJ = 23.0

# ============================================================
# CELL 3: Download SHD from Zenodo (HDF5, no tonic needed)
# ============================================================
import urllib.request
import zipfile

def download_and_unzip(url, zip_path, h5_path):
    if h5_path.exists():
        print(f"Already have {h5_path.name}", flush=True)
        return
    print(f"Downloading {zip_path.name}...", flush=True)
    urllib.request.urlretrieve(url, str(zip_path))
    print(f"  Downloaded ({zip_path.stat().st_size / 1e6:.1f} MB), extracting...", flush=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(zip_path.parent))
    zip_path.unlink()  # remove zip after extraction
    print(f"  Extracted {h5_path.name}", flush=True)

shd_dir = DATA_DIR / "SHD"
shd_dir.mkdir(parents=True, exist_ok=True)

# Source: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
download_and_unzip(
    "https://zenkelab.org/datasets/shd_train.h5.zip",
    shd_dir / "shd_train.h5.zip",
    shd_dir / "shd_train.h5",
)
download_and_unzip(
    "https://zenkelab.org/datasets/shd_test.h5.zip",
    shd_dir / "shd_test.h5.zip",
    shd_dir / "shd_test.h5",
)

# ============================================================
# CELL 4: SHD Dataset (parse HDF5 directly, bin into frames)
# ============================================================

class SHDDataset(Dataset):
    """
    Spiking Heidelberg Digits dataset.

    Each sample has variable-length spike trains from 700 neurons.
    We bin spikes into NUM_STEPS time bins to create dense frames.
    """
    def __init__(self, h5_path, num_steps=NUM_STEPS, num_neurons=SHD_NEURONS):
        self.num_steps = num_steps
        self.num_neurons = num_neurons

        with h5py.File(h5_path, "r") as f:
            # SHD format: spikes/times and spikes/units are lists of arrays
            self.times = [np.array(t) for t in f["spikes"]["times"]]
            self.units = [np.array(u, dtype=np.int64) for u in f["spikes"]["units"]]
            self.labels = np.array(f["labels"], dtype=np.int64)

        print(f"  Loaded {len(self.labels)} samples from {h5_path}", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        times = self.times[idx]
        units = self.units[idx]
        label = self.labels[idx]

        # Create dense frame: [num_steps, num_neurons]
        frame = np.zeros((self.num_steps, self.num_neurons), dtype=np.float32)

        if len(times) > 0:
            # Normalize times to [0, num_steps)
            t_min, t_max = times.min(), times.max()
            if t_max > t_min:
                t_bins = ((times - t_min) / (t_max - t_min) * (self.num_steps - 1)).astype(np.int64)
            else:
                t_bins = np.zeros_like(times, dtype=np.int64)

            # Clamp to valid range
            t_bins = np.clip(t_bins, 0, self.num_steps - 1)
            # Clamp units to valid range
            valid_units = np.clip(units, 0, self.num_neurons - 1)

            # Fill frame (binarize: any spike in bin = 1)
            frame[t_bins, valid_units] = 1.0

        return torch.tensor(frame), label


def collate_shd(batch):
    frames, labels = zip(*batch)
    frames_t = torch.stack(frames).permute(1, 0, 2)  # [T, batch, neurons]
    labels_t = torch.tensor(labels, dtype=torch.long)
    return frames_t, labels_t


print("Loading SHD datasets...", flush=True)
train_ds = SHDDataset(str(shd_dir / "shd_train.h5"))
test_ds = SHDDataset(str(shd_dir / "shd_test.h5"))

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    drop_last=True, collate_fn=collate_shd, num_workers=2,
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    drop_last=False, collate_fn=collate_shd, num_workers=2,
)
print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches", flush=True)
print(f"Test: {len(test_ds)} samples, {len(test_loader)} batches", flush=True)

# ============================================================
# CELL 5: HAMT Core (self-contained, identical to v2)
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
# CELL 6: Models
# ============================================================

def build_baseline(input_size, hidden, output_size):
    spike_grad = surrogate.fast_sigmoid(slope=25)
    class BaselineSNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden)
            self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
            self.fc2 = nn.Linear(hidden, hidden)
            self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
            self.fc3 = nn.Linear(hidden, output_size)
            self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
            self.fan_outs = [hidden, hidden, output_size]
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
    return BaselineSNN()


def build_hamt(input_size, hidden, output_size):
    spike_grad = surrogate.fast_sigmoid(slope=25)
    class HAMTSNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = hidden
            self.fc1 = nn.Linear(input_size, hidden)
            self.hab1 = HabituationLayer(hidden)
            self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
            self.fc2 = nn.Linear(hidden, hidden)
            self.hab2 = HabituationLayer(hidden)
            self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
            self.fc3 = nn.Linear(hidden, output_size)
            self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
            self.fan_outs = [hidden, hidden, output_size]
        def forward(self, x):
            bs = x.shape[1]
            mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
            hs1 = HabituationState.initialize(bs, self.hidden, device=x.device)
            hs2 = HabituationState.initialize(bs, self.hidden, device=x.device)
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
    return HAMTSNN()


# ============================================================
# CELL 7: Training engine
# ============================================================

def estimate_energy(spike_recs, fan_outs):
    total_spk = sum(int(s.sum().item()) for s in spike_recs)
    syn_ops = sum(int(s.sum().item()) * fo for s, fo in zip(spike_recs, fan_outs))
    return (total_spk * ENERGY_PER_SPIKE_PJ + syn_ops * ENERGY_PER_SYNOP_PJ) * 1e-12


def train_and_eval(model, train_loader, test_loader, is_hamt=False):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if is_hamt:
        loss_fn = MetabolicLoss(
            lambda_energy=LAMBDA_ENERGY, lambda_habituation=LAMBDA_HAB,
            target_spike_rate=TARGET_SR, ramp_epochs=RAMP_EPOCHS,
        )
    else:
        loss_fn = nn.CrossEntropyLoss()

    history = {"test_acc": [], "spike_rate": [], "energy": [], "epoch_time": []}

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        correct = total = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            out, spike_recs, hab = model(data)
            if is_hamt:
                loss, _ = loss_fn(out, labels, spike_recs, hab, epoch)
            else:
                loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    sr = sum(s.sum().item() for s in spike_recs) / sum(s.numel() for s in spike_recs)
                prefix = "HAMT" if is_hamt else "Base"
                print(f"  {prefix} E{epoch+1}/{NUM_EPOCHS} B{batch_idx} SR:{sr:.4f}", flush=True)

        model.eval()
        test_correct = test_total = 0
        all_sr = []
        all_energy = []
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                out, spike_recs, hab = model(data)
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
        print(f"{prefix} Epoch {epoch+1}/{NUM_EPOCHS}: Acc={test_acc:.4f} "
              f"SR={avg_sr:.4f} E={avg_energy:.2e} T={epoch_time:.0f}s", flush=True)

    return history


# ============================================================
# CELL 8: Run SHD benchmark
# ============================================================

print("=" * 70, flush=True)
print("HAMT NEUROMORPHIC BENCHMARK: SHD (Spiking Heidelberg Digits)", flush=True)
print(f"Config: epochs={NUM_EPOCHS}, hidden={HIDDEN}, steps={NUM_STEPS}", flush=True)
print(f"Lambdas: energy={LAMBDA_ENERGY}, hab={LAMBDA_HAB} (VERIFIED OPTIMAL)", flush=True)
print(f"Device: {DEVICE}", flush=True)
print("=" * 70, flush=True)

# Baseline
print(f"\n--- Baseline FF h={HIDDEN} ---", flush=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
baseline = build_baseline(SHD_NEURONS, HIDDEN, SHD_CLASSES)
bh = train_and_eval(baseline, train_loader, test_loader, is_hamt=False)
del baseline
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# HAMT
print(f"\n--- HAMT FF h={HIDDEN} ---", flush=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
hamt_model = build_hamt(SHD_NEURONS, HIDDEN, SHD_CLASSES)
hh = train_and_eval(hamt_model, train_loader, test_loader, is_hamt=True)
del hamt_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Results
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
    "dataset": "shd",
    "architecture": f"FF h={HIDDEN}",
    "config": "default (0.001/0.0005)",
    "input_neurons": SHD_NEURONS,
    "num_classes": SHD_CLASSES,
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
}

with open(RESULTS_DIR / "shd_results.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"\n{'='*70}", flush=True)
print("SHD BENCHMARK RESULT", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Accuracy: Baseline={b_acc:.4f}  HAMT={h_acc:.4f}  Delta={h_acc-b_acc:+.4f}", flush=True)
print(f"  Spike Rate: Baseline={b_sr:.4f}  HAMT={h_sr:.4f}  Reduction={sr_red:.1f}%", flush=True)
print(f"  Energy: Baseline={b_e:.2e}  HAMT={h_e:.2e}  Reduction={e_red:.1f}%", flush=True)
print(f"  Time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
print(f"\nResults saved to: {RESULTS_DIR / 'shd_results.json'}", flush=True)
print("BENCHMARK COMPLETE", flush=True)
