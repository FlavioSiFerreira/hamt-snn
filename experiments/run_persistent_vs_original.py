"""
Compare Original vs Persistent Habituation on Fashion-MNIST.

Tests whether cross-sample memory makes habituation compound over
training epochs (the gap should GROW, not stay proportional).

Runs 3 conditions for 50 epochs each:
  1. Baseline (no HAMT)
  2. Original HAMT (within-sample habituation only)
  3. Persistent HAMT (cross-sample memory)

50 epochs is enough to see if the curves diverge.
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

from src.habituation.habituation_module import HabituationLayer, HabituationState
from src.habituation.persistent_habituation import PersistentHabituationLayer
from src.losses.metabolic_loss import MetabolicLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "persistent_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_ENERGY = 0.001
LAMBDA_HAB = 0.0005
TARGET_SR = 0.05
RAMP_EPOCHS = 10
NUM_STEPS = 25
HIDDEN = 800
NUM_EPOCHS = 50
BATCH_SIZE = 128
LR = 5e-4
SEED = 42

ENERGY_PER_SPIKE_PJ = 120.0
ENERGY_PER_SYNOP_PJ = 23.0


def estimate_energy(spike_recs, fan_outs):
    total_spk = sum(int(s.sum().item()) for s in spike_recs)
    syn_ops = sum(int(s.sum().item()) * fo for s, fo in zip(spike_recs, fan_outs))
    return (total_spk * ENERGY_PER_SPIKE_PJ + syn_ops * ENERGY_PER_SYNOP_PJ) * 1e-12


def rate_encode(images, num_steps):
    flat = images.view(images.shape[0], -1).clamp(0.0, 1.0)
    return (torch.rand(num_steps, flat.shape[0], flat.shape[1], device=flat.device) < flat.unsqueeze(0)).float()


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


class OriginalHAMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = HIDDEN
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


class PersistentHAMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = HIDDEN
        self.fc1 = nn.Linear(784, HIDDEN)
        self.hab1 = PersistentHabituationLayer(
            HIDDEN, memory_decay=0.999, long_term_weight=0.3, warmup_batches=50
        )
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.hab2 = PersistentHabituationLayer(
            HIDDEN, memory_decay=0.999, long_term_weight=0.3, warmup_batches=50
        )
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
        self.fan_outs = [HIDDEN, HIDDEN, 10]

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


def train_model(model, train_loader, test_loader, is_hamt, label):
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
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out, spike_recs, hab = model(rate_encode(imgs, NUM_STEPS))
            if is_hamt:
                loss, _ = loss_fn(out, labels, spike_recs, hab, epoch)
            else:
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
                out, spike_recs, hab = model(rate_encode(imgs, NUM_STEPS))
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
            mem_info = ""
            if hasattr(model, 'hab1') and hasattr(model.hab1, 'get_memory_stats'):
                stats = model.hab1.get_memory_stats()
                mem_info = f" LTM={stats['long_term_active']} batches={stats['batch_count']}"
            print(f"{label} Epoch {epoch+1}/{NUM_EPOCHS}: "
                  f"Acc={test_acc:.4f} SR={avg_sr:.4f} E={avg_energy:.2e}{mem_info}",
                  flush=True)

    return history


def main():
    print(f"[START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Experiment: Original vs Persistent HAMT, {NUM_EPOCHS} epochs", flush=True)
    print(f"Config: h={HIDDEN}, T={NUM_STEPS}, lr={LR}", flush=True)
    print("=" * 70, flush=True)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    data_dir = str(PROJECT_ROOT / "data")
    train_ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    all_results = {}

    # Condition 1: Baseline
    print(f"\n{'='*70}", flush=True)
    print("Condition 1: BASELINE (no HAMT)", flush=True)
    print(f"{'='*70}", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    bh = train_model(BaselineSNN(), train_loader, test_loader, False, "Base")
    all_results["baseline"] = bh
    torch.cuda.empty_cache()

    # Condition 2: Original HAMT
    print(f"\n{'='*70}", flush=True)
    print("Condition 2: ORIGINAL HAMT (within-sample only)", flush=True)
    print(f"{'='*70}", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    oh = train_model(OriginalHAMT(), train_loader, test_loader, True, "Orig")
    all_results["original_hamt"] = oh
    torch.cuda.empty_cache()

    # Condition 3: Persistent HAMT
    print(f"\n{'='*70}", flush=True)
    print("Condition 3: PERSISTENT HAMT (cross-sample memory)", flush=True)
    print(f"{'='*70}", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    ph = train_model(PersistentHAMT(), train_loader, test_loader, True, "Pers")
    all_results["persistent_hamt"] = ph
    torch.cuda.empty_cache()

    # Summary at milestones
    print(f"\n{'='*70}", flush=True)
    print("COMPARISON: Energy Reduction vs Baseline", flush=True)
    print(f"{'='*70}", flush=True)

    milestones = [5, 10, 15, 25, 50]
    print(f"\n{'Epoch':>6} {'Base E':>10} {'Orig E':>10} {'Orig Red%':>10} "
          f"{'Pers E':>10} {'Pers Red%':>10}", flush=True)
    print("-" * 60, flush=True)

    for ep in milestones:
        idx = ep - 1
        if idx < len(bh["energy"]):
            b_e = bh["energy"][idx]
            o_e = oh["energy"][idx]
            p_e = ph["energy"][idx]
            o_red = ((b_e - o_e) / max(b_e, 1e-10)) * 100
            p_red = ((b_e - p_e) / max(b_e, 1e-10)) * 100
            print(f"{ep:>6} {b_e:>10.2e} {o_e:>10.2e} {o_red:>+10.1f} "
                  f"{p_e:>10.2e} {p_red:>+10.1f}", flush=True)

    # Save full results
    with open(RESULTS_DIR / "persistent_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_DIR / 'persistent_comparison.json'}", flush=True)
    print(f"[DONE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
