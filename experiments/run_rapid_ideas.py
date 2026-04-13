"""
Rapid Idea Testing: 10 epochs each on MNIST, skip baseline (reuse known values).

Known MNIST baseline (from ablation, 15 epochs):
  Acc=98.02%, SR=4.22%, E=3.94e-03

We only care about ONE thing: does energy reduction GROW over epochs?
If the gap between baseline and the idea widens from epoch 1 to 10,
the approach has compounding potential. If flat, kill it.

Ideas tested:
  A) Threshold Adaptation: neurons that fire a lot get harder to fire
  B) Spike Budget: each neuron has a finite energy budget that depletes
  C) Weight Decay Proportional to Firing: high-firing neurons' weights shrink
  D) Competitive Inhibition: only top-K% of neurons fire per timestep
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
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "rapid_ideas"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_STEPS = 25
HIDDEN = 800
NUM_EPOCHS = 10
BATCH_SIZE = 128
LR = 5e-4
SEED = 42

ENERGY_PER_SPIKE_PJ = 120.0
ENERGY_PER_SYNOP_PJ = 23.0

spike_grad = surrogate.fast_sigmoid(slope=25)

# Surrogate spike function for manual LIF implementations
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, threshold):
        spk = (mem > threshold).float()
        ctx.save_for_backward(mem, threshold if isinstance(threshold, torch.Tensor) else torch.tensor(threshold))
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        mem, threshold = ctx.saved_tensors
        # Fast sigmoid surrogate gradient
        grad = 25 * torch.sigmoid(25 * (mem - threshold)) * (1 - torch.sigmoid(25 * (mem - threshold)))
        return grad_output * grad, None

def surrogate_spike(mem, threshold):
    if isinstance(threshold, (int, float)):
        threshold = torch.tensor(threshold, device=mem.device)
    return SurrogateSpike.apply(mem, threshold)


def estimate_energy(spike_recs, fan_outs):
    total_spk = sum(int(s.sum().item()) for s in spike_recs)
    syn_ops = sum(int(s.sum().item()) * fo for s, fo in zip(spike_recs, fan_outs))
    return (total_spk * ENERGY_PER_SPIKE_PJ + syn_ops * ENERGY_PER_SYNOP_PJ) * 1e-12


def rate_encode(images, num_steps):
    flat = images.view(images.shape[0], -1).clamp(0.0, 1.0)
    return (torch.rand(num_steps, flat.shape[0], flat.shape[1], device=flat.device) < flat.unsqueeze(0)).float()


# ============================================================
# IDEA A: Adaptive Threshold
# Neurons that fire frequently get their threshold raised.
# Like biological intrinsic plasticity.
# ============================================================

class AdaptiveThresholdSNN(nn.Module):
    """
    After each forward pass, neurons that fired above average
    get their threshold permanently raised. Over training,
    chatty neurons become progressively harder to activate.
    """
    def __init__(self, threshold_growth=0.01):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.fan_outs = [HIDDEN, HIDDEN, 10]
        self.threshold_growth = threshold_growth

        # Persistent adaptive thresholds (start at 1.0, grow for active neurons)
        self.register_buffer('thresh1', torch.ones(HIDDEN))
        self.register_buffer('thresh2', torch.ones(HIDDEN))

    def forward(self, x):
        beta = 0.95
        mem1 = torch.zeros(x.shape[1], HIDDEN, device=x.device)
        mem2 = torch.zeros(x.shape[1], HIDDEN, device=x.device)
        mem3 = torch.zeros(x.shape[1], 10, device=x.device)
        s1r, s2r, s3r = [], [], []

        for t in range(x.shape[0]):
            # Layer 1: manual LIF with adaptive threshold
            mem1 = beta * mem1 + self.fc1(x[t])
            spk1 = surrogate_spike(mem1, self.thresh1)
            mem1 = mem1 * (1 - spk1)  # reset on spike
            s1r.append(spk1)

            # Layer 2: manual LIF with adaptive threshold
            mem2 = beta * mem2 + self.fc2(spk1)
            spk2 = surrogate_spike(mem2, self.thresh2)
            mem2 = mem2 * (1 - spk2)
            s2r.append(spk2)

            # Layer 3: standard (output, no threshold adaptation)
            mem3 = beta * mem3 + self.fc3(spk2)
            spk3 = surrogate_spike(mem3, 1.0)
            mem3 = mem3 * (1 - spk3)
            s3r.append(spk3)

        s1r, s2r, s3r = torch.stack(s1r), torch.stack(s2r), torch.stack(s3r)

        # Store firing rates for post-backward threshold update
        if self.training:
            self._last_fr1 = s1r.detach().mean(dim=(0, 1))
            self._last_fr2 = s2r.detach().mean(dim=(0, 1))

        return s3r.sum(0), [s1r, s2r, s3r]

    def update_thresholds(self):
        """Call AFTER loss.backward() and optimizer.step()."""
        if hasattr(self, '_last_fr1'):
            with torch.no_grad():
                median1 = self._last_fr1.median()
                median2 = self._last_fr2.median()
                self.thresh1.add_(self.threshold_growth * (self._last_fr1 - median1).clamp(min=0))
                self.thresh2.add_(self.threshold_growth * (self._last_fr2 - median2).clamp(min=0))


# ============================================================
# IDEA B: Spike Budget
# Each neuron starts with a budget. Every spike costs 1.
# When budget runs low, the neuron's threshold rises sharply.
# Budget partially replenishes each epoch but shrinks over time.
# ============================================================

class SpikeBudgetSNN(nn.Module):
    """
    Neurons have a finite spike budget per epoch.
    As they approach their budget limit, firing becomes harder.
    Budget capacity shrinks 5% each epoch (compounding pressure).
    """
    def __init__(self, initial_budget=5000, budget_decay=0.95):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.fan_outs = [HIDDEN, HIDDEN, 10]
        self.initial_budget = initial_budget
        self.budget_decay = budget_decay

        self.register_buffer('budget1', torch.full((HIDDEN,), float(initial_budget)))
        self.register_buffer('budget2', torch.full((HIDDEN,), float(initial_budget)))
        self.register_buffer('spent1', torch.zeros(HIDDEN))
        self.register_buffer('spent2', torch.zeros(HIDDEN))
        self.register_buffer('max_budget', torch.tensor(float(initial_budget)))

    def forward(self, x):
        beta = 0.95
        mem1 = torch.zeros(x.shape[1], HIDDEN, device=x.device)
        mem2 = torch.zeros(x.shape[1], HIDDEN, device=x.device)
        mem3 = torch.zeros(x.shape[1], 10, device=x.device)
        s1r, s2r, s3r = [], [], []

        # Compute dynamic threshold based on remaining budget
        # As budget depletes, threshold rises exponentially
        budget_ratio1 = (self.budget1 / self.max_budget).clamp(0.01, 1.0)
        budget_ratio2 = (self.budget2 / self.max_budget).clamp(0.01, 1.0)
        thresh1 = 1.0 / budget_ratio1  # low budget = high threshold
        thresh2 = 1.0 / budget_ratio2

        for t in range(x.shape[0]):
            mem1 = beta * mem1 + self.fc1(x[t])
            spk1 = surrogate_spike(mem1, thresh1)
            mem1 = mem1 * (1 - spk1)
            s1r.append(spk1)

            mem2 = beta * mem2 + self.fc2(spk1)
            spk2 = surrogate_spike(mem2, thresh2)
            mem2 = mem2 * (1 - spk2)
            s2r.append(spk2)

            mem3 = beta * mem3 + self.fc3(spk2)
            spk3 = surrogate_spike(mem3, 1.0)
            mem3 = mem3 * (1 - spk3)
            s3r.append(spk3)

        s1r, s2r, s3r = torch.stack(s1r), torch.stack(s2r), torch.stack(s3r)

        # Track spending
        if self.training:
            with torch.no_grad():
                batch_spikes1 = s1r.sum(dim=(0, 1))
                batch_spikes2 = s2r.sum(dim=(0, 1))
                self.spent1 += batch_spikes1
                self.spent2 += batch_spikes2
                self.budget1 = (self.max_budget - self.spent1).clamp(min=0)
                self.budget2 = (self.max_budget - self.spent2).clamp(min=0)

        return s3r.sum(0), [s1r, s2r, s3r]

    def reset_epoch(self):
        """Call at start of each epoch. Budget shrinks over time."""
        with torch.no_grad():
            self.max_budget *= self.budget_decay
            self.spent1.zero_()
            self.spent2.zero_()
            self.budget1.fill_(self.max_budget.item())
            self.budget2.fill_(self.max_budget.item())


# ============================================================
# IDEA C: Activity-Proportional Weight Decay
# Weights connected to high-firing neurons decay faster.
# Like synaptic depression but applied through optimizer.
# ============================================================

class ActivityDecaySNN(nn.Module):
    """
    Standard SNN but after each batch, weights connected to
    high-firing neurons get extra L2 decay. Effectively shrinks
    pathways that carry redundant (familiar) information.
    """
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

        # Track cumulative firing rates
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
        """Call after optimizer.step(). Decays weights of chatty neurons."""
        if self.batch_count == 0:
            return
        with torch.no_grad():
            avg_fire1 = self.cum_fire1 / self.batch_count
            avg_fire2 = self.cum_fire2 / self.batch_count

            # Neurons firing above median get weight decay
            mask1 = (avg_fire1 > avg_fire1.median()).float()
            mask2 = (avg_fire2 > avg_fire2.median()).float()

            # Decay outgoing weights of high-firing neurons
            self.fc1.weight.data *= 1 - self.decay_rate * mask1.unsqueeze(1).expand_as(self.fc1.weight)
            self.fc2.weight.data *= 1 - self.decay_rate * mask2.unsqueeze(1).expand_as(self.fc2.weight)


# ============================================================
# IDEA D: Winner-Take-All (top-K sparsity)
# Only the top K% most active neurons fire per timestep.
# Forces extreme sparsity. K shrinks over training.
# ============================================================

class WinnerTakeAllSNN(nn.Module):
    """
    After computing membrane potential, only the top-K neurons
    (by membrane potential) are allowed to fire. K starts at 50%
    and shrinks to 10% over training, forcing increasing sparsity.
    """
    def __init__(self, initial_k_pct=0.5, final_k_pct=0.10):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.fan_outs = [HIDDEN, HIDDEN, 10]
        self.initial_k_pct = initial_k_pct
        self.final_k_pct = final_k_pct
        self.register_buffer('current_k_pct', torch.tensor(initial_k_pct))

    def _top_k_mask(self, mem, k_pct):
        k = max(1, int(mem.shape[1] * k_pct))
        _, topk_idx = torch.topk(mem, k, dim=1)
        mask = torch.zeros_like(mem)
        mask.scatter_(1, topk_idx, 1.0)
        return mask

    def forward(self, x):
        beta = 0.95
        mem1 = torch.zeros(x.shape[1], HIDDEN, device=x.device)
        mem2 = torch.zeros(x.shape[1], HIDDEN, device=x.device)
        mem3 = torch.zeros(x.shape[1], 10, device=x.device)
        s1r, s2r, s3r = [], [], []
        k = self.current_k_pct.item()

        for t in range(x.shape[0]):
            mem1 = beta * mem1 + self.fc1(x[t])
            # Only top-K neurons can fire
            can_fire1 = self._top_k_mask(mem1.detach(), k)
            spk1 = surrogate_spike(mem1, 1.0) * can_fire1
            mem1 = mem1 * (1 - spk1)
            s1r.append(spk1)

            mem2 = beta * mem2 + self.fc2(spk1)
            can_fire2 = self._top_k_mask(mem2.detach(), k)
            spk2 = surrogate_spike(mem2, 1.0) * can_fire2
            mem2 = mem2 * (1 - spk2)
            s2r.append(spk2)

            mem3 = beta * mem3 + self.fc3(spk2)
            spk3 = surrogate_spike(mem3, 1.0)
            mem3 = mem3 * (1 - spk3)
            s3r.append(spk3)

        s1r, s2r, s3r = torch.stack(s1r), torch.stack(s2r), torch.stack(s3r)
        return s3r.sum(0), [s1r, s2r, s3r]

    def update_k(self, epoch, total_epochs):
        """Shrink K linearly from initial to final over training."""
        progress = epoch / max(total_epochs - 1, 1)
        new_k = self.initial_k_pct - progress * (self.initial_k_pct - self.final_k_pct)
        self.current_k_pct.fill_(new_k)


# ============================================================
# Training loop (shared, no baseline needed)
# ============================================================

def train_idea(model, train_loader, test_loader, label, special_fn=None):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = {"test_acc": [], "spike_rate": [], "energy": [], "epoch_time": []}

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        # Per-epoch hooks
        if special_fn == "budget_reset" and hasattr(model, 'reset_epoch'):
            model.reset_epoch()
        if special_fn == "wta_update" and hasattr(model, 'update_k'):
            model.update_k(epoch, NUM_EPOCHS)

        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out, spike_recs = model(rate_encode(imgs, NUM_STEPS))
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Post-step hooks
            if hasattr(model, 'update_thresholds'):
                model.update_thresholds()
            if special_fn == "activity_decay" and hasattr(model, 'apply_activity_decay'):
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

        print(f"  {label} E{epoch+1}: Acc={test_acc:.4f} SR={avg_sr:.4f} "
              f"E={avg_energy:.2e} ({epoch_time:.0f}s)", flush=True)

    return history


# ============================================================
# Main
# ============================================================

def main():
    print(f"[START] {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"RAPID IDEA TESTING: 10 epochs each, MNIST", flush=True)
    print(f"Baseline reference: Acc=98.02%, SR=4.22%, E=3.94e-03", flush=True)
    print("=" * 60, flush=True)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    data_dir = str(PROJECT_ROOT / "data")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    results = {}

    ideas = [
        ("A_adaptive_thresh", AdaptiveThresholdSNN(threshold_growth=0.01), None),
        ("B_spike_budget", SpikeBudgetSNN(initial_budget=5000, budget_decay=0.95), "budget_reset"),
        ("C_activity_decay", ActivityDecaySNN(decay_rate=0.001), "activity_decay"),
        ("D_winner_take_all", WinnerTakeAllSNN(initial_k_pct=0.5, final_k_pct=0.10), "wta_update"),
    ]

    for name, model, special in ideas:
        print(f"\n{'='*60}", flush=True)
        print(f"IDEA {name}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            h = train_idea(model, train_loader, test_loader, name[:6], special)
            results[name] = h
            torch.cuda.empty_cache()

            e1 = h["energy"][0]
            e10 = h["energy"][-1]
            acc = max(h["test_acc"])
            trend = "GROWING" if e10 < e1 * 0.9 else ("STABLE" if e10 < e1 * 1.05 else "WORSENING")
            print(f"  VERDICT: E1={e1:.2e} E10={e10:.2e} trend={trend} bestAcc={acc:.4f}", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("RAPID TESTING SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Baseline ref: E=3.94e-03, Acc=98.02%", flush=True)
    print(f"\n{'Idea':<25} {'E(ep1)':>10} {'E(ep10)':>10} {'Trend':>10} {'BestAcc':>8}", flush=True)
    print("-" * 65, flush=True)
    for name, h in results.items():
        e1 = h["energy"][0]
        e10 = h["energy"][-1]
        trend = "GROWING" if e10 < e1 * 0.9 else ("STABLE" if e10 < e1 * 1.05 else "WORSE")
        print(f"{name:<25} {e1:>10.2e} {e10:>10.2e} {trend:>10} {max(h['test_acc']):>8.4f}", flush=True)

    with open(RESULTS_DIR / "rapid_ideas_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[DONE] {datetime.now().strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
