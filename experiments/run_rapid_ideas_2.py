"""
Rapid Idea Testing Round 2: 10 epochs each on MNIST.

E) Predictive Coding: each layer predicts the next layer's input.
   Only prediction errors get transmitted as spikes.
F) Information Bottleneck: force hidden through 100 neurons then expand back.

Baseline ref: Acc=98.02%, E=3.94e-03
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


def estimate_energy(spike_recs, fan_outs):
    total_spk = sum(int(s.sum().item()) for s in spike_recs)
    syn_ops = sum(int(s.sum().item()) * fo for s, fo in zip(spike_recs, fan_outs))
    return (total_spk * ENERGY_PER_SPIKE_PJ + syn_ops * ENERGY_PER_SYNOP_PJ) * 1e-12


def rate_encode(images, num_steps):
    flat = images.view(images.shape[0], -1).clamp(0.0, 1.0)
    return (torch.rand(num_steps, flat.shape[0], flat.shape[1], device=flat.device) < flat.unsqueeze(0)).float()


# ============================================================
# IDEA E: Predictive Coding SNN
# Each layer learns to predict the input it will receive.
# Only the prediction error (residual) gets passed to the neuron.
# Prediction improves over training, so fewer spikes needed.
# ============================================================

class PredictiveCodingSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
        self.fan_outs = [HIDDEN, HIDDEN, 10]

        # Prediction layers: predict the post-linear, pre-spike activation
        # from the previous timestep's spike output
        self.pred1 = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.pred2 = nn.Linear(HIDDEN, HIDDEN, bias=False)

        # Initialize predictions to zero (no prediction initially)
        nn.init.zeros_(self.pred1.weight)
        nn.init.zeros_(self.pred2.weight)

    def forward(self, x):
        mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        s1r, s2r, s3r = [], [], []
        prev_spk1 = torch.zeros(x.shape[1], HIDDEN, device=x.device)
        prev_spk2 = torch.zeros(x.shape[1], HIDDEN, device=x.device)

        for t in range(x.shape[0]):
            # Layer 1: compute activation, subtract prediction, spike on error
            activation1 = self.fc1(x[t])
            prediction1 = self.pred1(prev_spk1)
            error1 = activation1 - prediction1  # only error gets transmitted
            spk1, mem1 = self.lif1(error1, mem1)
            prev_spk1 = spk1.detach()
            s1r.append(spk1)

            # Layer 2: same pattern
            activation2 = self.fc2(spk1)
            prediction2 = self.pred2(prev_spk2)
            error2 = activation2 - prediction2
            spk2, mem2 = self.lif2(error2, mem2)
            prev_spk2 = spk2.detach()
            s2r.append(spk2)

            # Layer 3: no prediction (output layer)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            s3r.append(spk3)

        s1r, s2r, s3r = torch.stack(s1r), torch.stack(s2r), torch.stack(s3r)
        return s3r.sum(0), [s1r, s2r, s3r]


# ============================================================
# IDEA F: Information Bottleneck SNN
# Force representation through a narrow bottleneck (100 neurons).
# Fewer neurons = fewer possible spikes = less energy by design.
# Then expand back for classification.
# ============================================================

class BottleneckSNN(nn.Module):
    def __init__(self, bottleneck=100):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        # Compress to bottleneck
        self.fc_compress = nn.Linear(HIDDEN, bottleneck)
        self.lif_compress = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        # Expand back
        self.fc_expand = nn.Linear(bottleneck, HIDDEN)
        self.lif_expand = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
        self.fan_outs = [HIDDEN, bottleneck, HIDDEN, 10]
        self.bottleneck = bottleneck

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem_c = self.lif_compress.init_leaky()
        mem_e = self.lif_expand.init_leaky()
        mem3 = self.lif3.init_leaky()
        s1r, scr, ser, s3r = [], [], [], []

        for t in range(x.shape[0]):
            spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
            s1r.append(spk1)

            spk_c, mem_c = self.lif_compress(self.fc_compress(spk1), mem_c)
            scr.append(spk_c)

            spk_e, mem_e = self.lif_expand(self.fc_expand(spk_c), mem_e)
            ser.append(spk_e)

            spk3, mem3 = self.lif3(self.fc3(spk_e), mem3)
            s3r.append(spk3)

        s1r = torch.stack(s1r)
        scr = torch.stack(scr)
        ser = torch.stack(ser)
        s3r = torch.stack(s3r)
        return s3r.sum(0), [s1r, scr, ser, s3r]


# ============================================================
# Training
# ============================================================

def train_idea(model, train_loader, test_loader, label):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = {"test_acc": [], "spike_rate": [], "energy": [], "epoch_time": []}

    for epoch in range(NUM_EPOCHS):
        import time
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

        print(f"  {label} E{epoch+1}: Acc={test_acc:.4f} SR={avg_sr:.4f} "
              f"E={avg_energy:.2e} ({epoch_time:.0f}s)", flush=True)

    return history


def main():
    print(f"[START] {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"RAPID IDEAS ROUND 2: 10 epochs, MNIST", flush=True)
    print(f"Baseline ref: Acc=98.02%, E=3.94e-03", flush=True)
    print("=" * 60, flush=True)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    data_dir = str(PROJECT_ROOT / "data")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    results = {}

    ideas = [
        ("E_predictive_coding", PredictiveCodingSNN()),
        ("F_bottleneck_100", BottleneckSNN(bottleneck=100)),
    ]

    for name, model in ideas:
        print(f"\n{'='*60}", flush=True)
        print(f"IDEA {name}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            h = train_idea(model, train_loader, test_loader, name[:6])
            results[name] = h
            torch.cuda.empty_cache()

            e1 = h["energy"][0]
            e10 = h["energy"][-1]
            acc = max(h["test_acc"])
            trend = "GROWING" if e10 < e1 * 0.9 else ("STABLE" if e10 < e1 * 1.05 else "WORSE")
            print(f"  VERDICT: E1={e1:.2e} E10={e10:.2e} trend={trend} bestAcc={acc:.4f}", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("ROUND 2 SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Baseline ref: E=3.94e-03, Acc=98.02%", flush=True)
    for name, h in results.items():
        e1 = h["energy"][0]
        e10 = h["energy"][-1]
        trend = "GROWING" if e10 < e1 * 0.9 else ("STABLE" if e10 < e1 * 1.05 else "WORSE")
        red = ((3.94e-03 - e10) / 3.94e-03) * 100
        print(f"  {name}: E10={e10:.2e} ({red:+.1f}% vs base) "
              f"trend={trend} bestAcc={max(h['test_acc']):.4f}", flush=True)

    with open(RESULTS_DIR / "rapid_ideas_round2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[DONE] {datetime.now().strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
