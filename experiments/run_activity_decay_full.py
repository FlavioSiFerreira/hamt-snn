"""
Full test of Idea C: Activity-Proportional Weight Decay

Rapid screen showed: 52% energy reduction in 10 epochs, 97.48% accuracy.
This is the most promising approach found today.

Mechanism: After each optimizer step, weights connected to neurons that
fire above the median rate get extra L2 decay. High-firing pathways
physically shrink over time, forcing the network to use fewer, more
efficient pathways.

50 epochs on MNIST. No baseline needed (known: Acc=98.02%, E=3.94e-03).
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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "activity_decay"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_STEPS = 25
HIDDEN = 800
NUM_EPOCHS = 50
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

    def get_stats(self):
        if self.batch_count == 0:
            return {}
        avg1 = self.cum_fire1 / self.batch_count
        avg2 = self.cum_fire2 / self.batch_count
        return {
            "above_median1": (avg1 > avg1.median()).float().mean().item() * 100,
            "max_fire1": avg1.max().item(),
            "min_fire1": avg1.min().item(),
            "w1_mean": self.fc1.weight.data.abs().mean().item(),
            "w2_mean": self.fc2.weight.data.abs().mean().item(),
        }


def main():
    print(f"[START] {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"IDEA C FULL TEST: Activity Decay, {NUM_EPOCHS} epochs, MNIST", flush=True)
    print(f"Baseline ref: Acc=98.02%, E=3.94e-03", flush=True)
    print("=" * 60, flush=True)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    data_dir = str(PROJECT_ROOT / "data")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = ActivityDecaySNN(decay_rate=0.001).to(DEVICE)
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

        vs_base = ((3.94e-03 - avg_energy) / 3.94e-03) * 100

        stats = model.get_stats()
        w_info = f" w1={stats.get('w1_mean', 0):.4f} w2={stats.get('w2_mean', 0):.4f}" if stats else ""

        print(f"E{epoch+1}/{NUM_EPOCHS}: Acc={test_acc:.4f} SR={avg_sr:.4f} "
              f"E={avg_energy:.2e} vsBase={vs_base:+.1f}%{w_info}", flush=True)

    # Summary at milestones
    print(f"\n{'='*60}", flush=True)
    print("ACTIVITY DECAY 50-EPOCH RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for ep in [1, 5, 10, 25, 50]:
        idx = ep - 1
        e = history["energy"][idx]
        red = ((3.94e-03 - e) / 3.94e-03) * 100
        print(f"  Epoch {ep:>3}: Acc={history['test_acc'][idx]:.4f} "
              f"E={e:.2e} vs baseline: {red:+.1f}%", flush=True)

    e1 = history["energy"][0]
    e50 = history["energy"][-1]
    print(f"\n  Energy E1={e1:.2e} -> E50={e50:.2e}", flush=True)
    if e50 < e1 * 0.8:
        print(f"  COMPOUNDING: energy at epoch 50 is {(1-e50/e1)*100:.0f}% lower than epoch 1", flush=True)
    else:
        print(f"  NOT COMPOUNDING: ratio = {e50/e1:.2f}", flush=True)

    with open(RESULTS_DIR / "activity_decay_50ep.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\n[DONE] {datetime.now().strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
