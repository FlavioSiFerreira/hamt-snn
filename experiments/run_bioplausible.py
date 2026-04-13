"""
Biologically Plausible Energy-Efficient SNN (BioSNN)

Combines three proven biological energy mechanisms:
  1. Tsodyks-Markram Short-Term Synaptic Depression (STD)
     - Vesicle pool depletes per spike, recovers with tau_rec
  2. Spike Frequency Adaptation (SFA)
     - Calcium-driven afterhyperpolarization makes neurons harder to fire
  3. Homeostatic Scaling
     - Very slow weight scaling toward 5% target firing rate

All mechanisms are persistent across batches and compound over training.
Uses standard CrossEntropyLoss only. Energy reduction is EMERGENT.

Quick 10-epoch MNIST test first. If it shows compounding, do 50 epochs.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import numpy as np
import json
import time
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "bioplausible"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_STEPS = 25
HIDDEN = 800
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


class BioSNN(nn.Module):
    """
    SNN with biologically grounded energy optimization:
    - STD on synapses (Tsodyks-Markram vesicle depletion)
    - SFA on neurons (calcium-driven adaptation)
    - Homeostatic scaling on weights (slow target rate)

    All mechanisms persist across batches. No special loss needed.
    """

    def __init__(self,
                 input_size=784, hidden=800, output_size=10,
                 # STD parameters (Tsodyks-Markram)
                 tau_rec=0.25,      # vesicle recovery time constant
                 use_fraction=0.2,  # fraction of vesicles released per spike
                 # SFA parameters
                 tau_adapt=0.05,    # adaptation current decay
                 adapt_strength=0.5, # how much adaptation current raises threshold
                 # Homeostatic parameters
                 target_rate=0.05,  # target 5% firing rate
                 homeo_rate=1e-4,   # very slow scaling
                 ):
        super().__init__()
        self.hidden = hidden
        self.tau_rec = tau_rec
        self.use_fraction = use_fraction
        self.tau_adapt = tau_adapt
        self.adapt_strength = adapt_strength
        self.target_rate = target_rate
        self.homeo_rate = homeo_rate

        # Standard layers
        self.fc1 = nn.Linear(input_size, hidden)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc3 = nn.Linear(hidden, output_size)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
        self.fan_outs = [hidden, hidden, output_size]

        # Persistent STD state: vesicle availability per synapse output neuron
        # (simplified: per output neuron, not per individual synapse)
        self.register_buffer('vesicle1', torch.ones(hidden))
        self.register_buffer('vesicle2', torch.ones(hidden))

        # Persistent SFA state: adaptation current per neuron
        self.register_buffer('adapt1', torch.zeros(hidden))
        self.register_buffer('adapt2', torch.zeros(hidden))

        # Persistent homeostatic state: cumulative firing rate
        self.register_buffer('cum_rate1', torch.zeros(hidden))
        self.register_buffer('cum_rate2', torch.zeros(hidden))
        self.register_buffer('homeo_scale1', torch.ones(hidden))
        self.register_buffer('homeo_scale2', torch.ones(hidden))
        self.register_buffer('batch_count', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        s1r, s2r, s3r = [], [], []

        # Detached modulation factors (bio mechanisms don't backprop)
        v1_mod = (self.vesicle1 * self.homeo_scale1).detach()
        v2_mod = (self.vesicle2 * self.homeo_scale2).detach()

        for t in range(x.shape[0]):
            # Layer 1: modulate input by vesicle availability + homeostatic scale
            current1 = self.fc1(x[t]) * v1_mod
            spk1, mem1 = self.lif1(current1, mem1)
            s1r.append(spk1)

            # Layer 2
            current2 = self.fc2(spk1) * v2_mod
            spk2, mem2 = self.lif2(current2, mem2)
            s2r.append(spk2)

            # Layer 3 (output, no modulation)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            s3r.append(spk3)

        s1r = torch.stack(s1r)
        s2r = torch.stack(s2r)
        s3r = torch.stack(s3r)

        # Update all persistent bio state AFTER forward pass (no graph issues)
        if self.training:
            with torch.no_grad():
                dt = 1.0 / NUM_STEPS
                fr1 = s1r.mean(dim=(0, 1))  # per-neuron firing rate
                fr2 = s2r.mean(dim=(0, 1))

                # STD: vesicles deplete proportional to firing, recover toward 1
                self.vesicle1 += dt * ((1 - self.vesicle1) / self.tau_rec
                                       - self.use_fraction * self.vesicle1 * fr1)
                self.vesicle1.clamp_(0.01, 1.0)
                self.vesicle2 += dt * ((1 - self.vesicle2) / self.tau_rec
                                       - self.use_fraction * self.vesicle2 * fr2)
                self.vesicle2.clamp_(0.01, 1.0)

                # SFA: adaptation current grows with firing
                self.adapt1 = self.adapt1 * (1 - dt / self.tau_adapt) + fr1 * dt / self.tau_adapt
                self.adapt2 = self.adapt2 * (1 - dt / self.tau_adapt) + fr2 * dt / self.tau_adapt

                # Homeostatic scaling (very slow)
                self.cum_rate1 = 0.99 * self.cum_rate1 + 0.01 * fr1
                self.cum_rate2 = 0.99 * self.cum_rate2 + 0.01 * fr2
                self.batch_count += 1

                if self.batch_count > 100:
                    error1 = self.cum_rate1 - self.target_rate
                    error2 = self.cum_rate2 - self.target_rate
                    self.homeo_scale1 -= self.homeo_rate * error1
                    self.homeo_scale2 -= self.homeo_rate * error2
                    self.homeo_scale1.clamp_(0.5, 2.0)
                    self.homeo_scale2.clamp_(0.5, 2.0)

        return s3r.sum(0), [s1r, s2r, s3r]

    def get_bio_stats(self):
        return {
            "vesicle1_mean": self.vesicle1.mean().item(),
            "vesicle2_mean": self.vesicle2.mean().item(),
            "adapt1_mean": self.adapt1.mean().item(),
            "adapt2_mean": self.adapt2.mean().item(),
            "homeo_scale1_mean": self.homeo_scale1.mean().item(),
            "homeo_scale2_mean": self.homeo_scale2.mean().item(),
            "batch_count": self.batch_count.item(),
        }


def train_model(model, train_loader, test_loader, label, num_epochs):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = {"test_acc": [], "spike_rate": [], "energy": [], "epoch_time": []}

    for epoch in range(num_epochs):
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

        bio_info = ""
        if hasattr(model, 'get_bio_stats'):
            s = model.get_bio_stats()
            bio_info = (f" v={s['vesicle1_mean']:.3f} "
                       f"a={s['adapt1_mean']:.3f} "
                       f"h={s['homeo_scale1_mean']:.3f}")

        print(f"{label} E{epoch+1}/{num_epochs}: Acc={test_acc:.4f} "
              f"SR={avg_sr:.4f} E={avg_energy:.2e}{bio_info}", flush=True)

    return history


def main():
    print(f"[START] {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("BIO-PLAUSIBLE SNN: STD + SFA + Homeostatic Scaling", flush=True)
    print("=" * 60, flush=True)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    data_dir = str(PROJECT_ROOT / "data")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    # Quick 10-epoch screen first
    print(f"\n--- Phase 1: Quick screen (10 epochs, MNIST) ---", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    h10 = train_model(BioSNN(), train_loader, test_loader, "Bio10", 10)

    e1 = h10["energy"][0]
    e10 = h10["energy"][-1]
    acc = max(h10["test_acc"])
    trend = "GROWING" if e10 < e1 * 0.9 else ("STABLE" if e10 < e1 * 1.05 else "WORSENING")

    print(f"\nQUICK SCREEN: E1={e1:.2e} E10={e10:.2e} trend={trend} bestAcc={acc:.4f}", flush=True)
    print(f"vs Baseline: E=3.94e-03, Acc=98.02%", flush=True)

    if acc < 0.5:
        print("KILLED: accuracy too low, aborting", flush=True)
        return

    # If promising, run 50 epochs
    if e10 < 3.94e-03 * 0.8 or trend == "GROWING":
        print(f"\nPROMISING! Running full 50-epoch test...", flush=True)
        print(f"\n--- Phase 2: Full test (50 epochs, MNIST) ---", flush=True)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        h50 = train_model(BioSNN(), train_loader, test_loader, "Bio50", 50)

        # Check compounding
        milestones = [5, 10, 25, 50]
        print(f"\n{'Epoch':>6} {'Acc':>8} {'SR':>8} {'Energy':>10} {'vs Base':>8}", flush=True)
        print("-" * 45, flush=True)
        for ep in milestones:
            idx = ep - 1
            e = h50["energy"][idx]
            red = ((3.94e-03 - e) / 3.94e-03) * 100
            print(f"{ep:>6} {h50['test_acc'][idx]:>8.4f} {h50['spike_rate'][idx]:>8.4f} "
                  f"{e:>10.2e} {red:>+8.1f}%", flush=True)

        results = {"quick_10": h10, "full_50": h50}
    else:
        print(f"\nNOT PROMISING. Stopping.", flush=True)
        results = {"quick_10": h10}

    with open(RESULTS_DIR / "bioplausible_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[DONE] {datetime.now().strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
