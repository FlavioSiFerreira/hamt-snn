"""
HAMT Neuromorphic Dataset Benchmark (Local GPU)

Tests HAMT on event-driven neuromorphic datasets where temporal
habituation should have a genuine advantage over rate-coded static images.

Datasets:
  - N-MNIST: Event-camera version of MNIST (native spikes, 34x34x2)
  - SHD: Spiking Heidelberg Digits (700 neurons, 20 classes, audio)

Config: Default HAMT (lambda_e=0.001, lambda_h=0.0005, h=800, T=25)
GPU: GTX 1650

Results saved to: experiments/results/neuromorphic/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import json
import time
from datetime import datetime

import tonic
import tonic.transforms as tonic_tf
from torch.utils.data import DataLoader

from src.habituation.habituation_module import HabituationLayer, HabituationState
from src.losses.metabolic_loss import MetabolicLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "neuromorphic"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Verified optimal config (feedback_hamt_hyperparams.md)
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

ENERGY_PER_SPIKE_PJ = 120.0
ENERGY_PER_SYNOP_PJ = 23.0


def estimate_energy(spike_recs, fan_outs):
    total_spk = sum(int(s.sum().item()) for s in spike_recs)
    syn_ops = sum(int(s.sum().item()) * fo for s, fo in zip(spike_recs, fan_outs))
    return (total_spk * ENERGY_PER_SPIKE_PJ + syn_ops * ENERGY_PER_SYNOP_PJ) * 1e-12


# ============================================================
# Dataset Loaders (Neuromorphic via Tonic)
# ============================================================

def load_nmnist(data_dir):
    """
    Load N-MNIST: event-camera MNIST, 34x34 sensor with 2 polarities.
    Convert events to dense frames binned over NUM_STEPS time bins.
    Flatten to [NUM_STEPS, 34*34*2] = [25, 2312] per sample.
    """
    sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)

    frame_transform = tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=NUM_STEPS),
    ])

    train_ds = tonic.datasets.NMNIST(
        save_to=str(data_dir), train=True, transform=frame_transform
    )
    test_ds = tonic.datasets.NMNIST(
        save_to=str(data_dir), train=False, transform=frame_transform
    )

    input_size = 34 * 34 * 2  # 2312

    def collate_fn(batch):
        frames, labels = zip(*batch)
        # frames: list of [T, C, H, W] arrays (from ToFrame)
        tensors = []
        for f in frames:
            t = torch.tensor(f, dtype=torch.float32)
            # Ensure exactly NUM_STEPS time bins
            if t.shape[0] < NUM_STEPS:
                pad = torch.zeros(NUM_STEPS - t.shape[0], *t.shape[1:])
                t = torch.cat([t, pad], dim=0)
            elif t.shape[0] > NUM_STEPS:
                t = t[:NUM_STEPS]
            # Flatten spatial + polarity: [T, C*H*W]
            t = t.reshape(NUM_STEPS, -1)
            # Binarize (these are event counts, convert to spikes)
            t = (t > 0).float()
            tensors.append(t)
        frames_t = torch.stack(tensors)  # [batch, T, features]
        frames_t = frames_t.permute(1, 0, 2)  # [T, batch, features]
        labels_t = torch.tensor(labels, dtype=torch.long)
        return frames_t, labels_t

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=False, collate_fn=collate_fn, num_workers=0
    )

    return train_loader, test_loader, input_size, 10


def load_shd(data_dir):
    """
    Load SHD: Spiking Heidelberg Digits.
    700 input neurons, 20 spoken digit classes.
    Events binned into NUM_STEPS time bins.
    """
    sensor_size = tonic.datasets.SHD.sensor_size  # (700,) or (1, 700) depending on version

    frame_transform = tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=NUM_STEPS),
    ])

    train_ds = tonic.datasets.SHD(
        save_to=str(data_dir), train=True, transform=frame_transform
    )
    test_ds = tonic.datasets.SHD(
        save_to=str(data_dir), train=False, transform=frame_transform
    )

    input_size = 700

    def collate_fn(batch):
        frames, labels = zip(*batch)
        tensors = []
        for f in frames:
            t = torch.tensor(f, dtype=torch.float32)
            if t.shape[0] < NUM_STEPS:
                pad = torch.zeros(NUM_STEPS - t.shape[0], *t.shape[1:])
                t = torch.cat([t, pad], dim=0)
            elif t.shape[0] > NUM_STEPS:
                t = t[:NUM_STEPS]
            # Flatten all dims except time
            t = t.reshape(NUM_STEPS, -1)
            # Take only first 700 features (in case of extra channel dim)
            if t.shape[1] > input_size:
                t = t[:, :input_size]
            # Binarize
            t = (t > 0).float()
            tensors.append(t)
        frames_t = torch.stack(tensors).permute(1, 0, 2)  # [T, batch, features]
        labels_t = torch.tensor(labels, dtype=torch.long)
        return frames_t, labels_t

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=False, collate_fn=collate_fn, num_workers=0
    )

    return train_loader, test_loader, input_size, 20


# ============================================================
# Models (FF with configurable input size)
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
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            s1r, s2r, s3r = [], [], []
            num_steps = x.shape[0]
            for t in range(num_steps):
                spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
                spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
                spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
                s1r.append(spk1)
                s2r.append(spk2)
                s3r.append(spk3)
            s1r = torch.stack(s1r)
            s2r = torch.stack(s2r)
            s3r = torch.stack(s3r)
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
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            hs1 = HabituationState.initialize(bs, self.hidden, device=x.device)
            hs2 = HabituationState.initialize(bs, self.hidden, device=x.device)
            s1r, s2r, s3r = [], [], []
            num_steps = x.shape[0]
            for t in range(num_steps):
                c1 = self.fc1(x[t])
                c1, hs1 = self.hab1(c1, hs1)
                spk1, mem1 = self.lif1(c1, mem1)
                c2 = self.fc2(spk1)
                c2, hs2 = self.hab2(c2, hs2)
                spk2, mem2 = self.lif2(c2, mem2)
                spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
                s1r.append(spk1)
                s2r.append(spk2)
                s3r.append(spk3)
            s1r = torch.stack(s1r)
            s2r = torch.stack(s2r)
            s3r = torch.stack(s3r)
            return s3r.sum(0), [s1r, s2r, s3r], [hs1.habituation_strength, hs2.habituation_strength]

    return HAMTSNN()


# ============================================================
# Training Engine
# ============================================================

def train_and_eval(model, train_loader, test_loader, is_hamt=False):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if is_hamt:
        loss_fn = MetabolicLoss(
            lambda_energy=LAMBDA_ENERGY,
            lambda_habituation=LAMBDA_HAB,
            target_spike_rate=TARGET_SR,
            ramp_epochs=RAMP_EPOCHS,
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

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    sr = sum(s.sum().item() for s in spike_recs) / sum(s.numel() for s in spike_recs)
                prefix = "HAMT" if is_hamt else "Base"
                print(f"  {prefix} E{epoch+1}/{NUM_EPOCHS} B{batch_idx} SR:{sr:.4f}", flush=True)

        # Test
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
# Run Benchmark
# ============================================================

def run_dataset(name, load_fn):
    print(f"\n{'='*70}", flush=True)
    print(f"DATASET: {name}", flush=True)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"{'='*70}", flush=True)

    data_dir = PROJECT_ROOT / "data" / "neuromorphic"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {name}...", flush=True)
    train_loader, test_loader, input_size, num_classes = load_fn(data_dir)
    print(f"Loaded: input={input_size}, classes={num_classes}, "
          f"train batches={len(train_loader)}, test batches={len(test_loader)}", flush=True)

    # Baseline
    print(f"\n--- Baseline FF h={HIDDEN} ---", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    baseline = build_baseline(input_size, HIDDEN, num_classes)
    bh = train_and_eval(baseline, train_loader, test_loader, is_hamt=False)
    del baseline
    torch.cuda.empty_cache()

    # HAMT
    print(f"\n--- HAMT FF h={HIDDEN} ---", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    hamt = build_hamt(input_size, HIDDEN, num_classes)
    hh = train_and_eval(hamt, train_loader, test_loader, is_hamt=True)
    del hamt
    torch.cuda.empty_cache()

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
        "dataset": name,
        "architecture": f"FF h={HIDDEN}",
        "config": "default (0.001/0.0005)",
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
        "num_epochs": NUM_EPOCHS,
        "num_steps": NUM_STEPS,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\nRESULT {name}:", flush=True)
    print(f"  Accuracy: B={b_acc:.4f} H={h_acc:.4f} (delta={h_acc-b_acc:+.4f})", flush=True)
    print(f"  Spike Rate: B={b_sr:.4f} H={h_sr:.4f} (reduction={sr_red:.1f}%)", flush=True)
    print(f"  Energy: B={b_e:.2e} H={h_e:.2e} (reduction={e_red:.1f}%)", flush=True)
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    return result


def main():
    print(f"[START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Config: lambda_e={LAMBDA_ENERGY}, lambda_h={LAMBDA_HAB}, "
          f"h={HIDDEN}, T={NUM_STEPS}, epochs={NUM_EPOCHS}", flush=True)

    all_results = {}

    # Load existing results if any
    results_file = RESULTS_DIR / "neuromorphic_results.json"
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results", flush=True)

    # Run datasets sequentially (max 2 at a time per self-improvement rule)
    datasets_to_run = [
        ("n_mnist", load_nmnist),
        ("shd", load_shd),
    ]

    for ds_name, load_fn in datasets_to_run:
        if ds_name in all_results:
            print(f"\nSkipping {ds_name} (already completed)", flush=True)
            continue

        result = run_dataset(ds_name, load_fn)
        all_results[ds_name] = result

        # Save after each dataset (checkpoint)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Checkpoint saved: {results_file}", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("NEUROMORPHIC BENCHMARK SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n{'Dataset':<15} {'B Acc':>8} {'H Acc':>8} {'Delta':>8} "
          f"{'E Red%':>8} {'SR Red%':>8}", flush=True)
    print("-" * 60, flush=True)
    for name, r in all_results.items():
        print(f"{name:<15} {r['baseline_best_acc']:>8.4f} {r['hamt_best_acc']:>8.4f} "
              f"{r['acc_delta']:>+8.4f} {r['energy_reduction_pct']:>+8.1f} "
              f"{r['spike_rate_reduction_pct']:>+8.1f}", flush=True)

    print(f"\n[DONE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Results: {results_file}", flush=True)


if __name__ == "__main__":
    main()
