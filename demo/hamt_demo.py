"""
HAMT Demo: 15-Minute Proof of Energy Savings

Run this on Google Colab (free GPU) or any machine with PyTorch.
It trains a baseline SNN and a HAMT-enhanced SNN on Fashion-MNIST,
then shows the energy reduction in a clear comparison.

Expected result: ~15-25% energy reduction, <1% accuracy difference.
Runtime: ~15-20 minutes on Colab T4 GPU.
"""

# Step 1: Install
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "snntorch", "-q"])

# Step 2: Setup
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# Step 3: HAMT module (self-contained, no external deps)
def _inv_sig(x):
    x = max(min(x, 0.999), 0.001)
    return -torch.log(torch.tensor(1.0 / x - 1.0)).item()

class HabState:
    __slots__ = ("fam", "hist", "strength")
    def __init__(self, fam, hist, strength):
        self.fam, self.hist, self.strength = fam, hist, strength
    @staticmethod
    def init(bs, n, hl=5, dev=torch.device("cpu")):
        return HabState(torch.zeros(bs,n,device=dev),
                        torch.zeros(hl,bs,n,device=dev),
                        torch.zeros(bs,n,device=dev))

class HabLayer(nn.Module):
    def __init__(self, n, alpha=0.5, tau_h=0.9, tau_r=0.8, thr=0.3):
        super().__init__()
        self.n, self.thr = n, thr
        self._a = nn.Parameter(torch.full((n,), _inv_sig(alpha)))
        self._th = nn.Parameter(torch.full((n,), _inv_sig(tau_h)))
        self._tr = nn.Parameter(torch.full((n,), _inv_sig(tau_r)))

    def forward(self, x, s):
        a, th, tr = torch.sigmoid(self._a), torch.sigmoid(self._th), torch.sigmoid(self._tr)
        nov = torch.abs(x - s.hist.mean(0)) / torch.max(torch.abs(x), torch.abs(s.hist.mean(0))).clamp(1e-8)
        fam_mask = (nov < self.thr).float()
        new_fam = (fam_mask * (th * s.fam + (1-th)) + (1-fam_mask) * (tr * s.fam)).clamp(0,1)
        new_str = a * new_fam
        out = x * (1.0 - new_str)
        new_hist = torch.cat([x.unsqueeze(0), s.hist[:-1]], 0)
        return out, HabState(new_fam, new_hist, new_str)

# Step 4: Models
SG = surrogate.fast_sigmoid(slope=25)
STEPS, HIDDEN = 25, 800

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.l1 = snn.Leaky(beta=0.95, spike_grad=SG, learn_beta=True)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.l2 = snn.Leaky(beta=0.95, spike_grad=SG, learn_beta=True)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.l3 = snn.Leaky(beta=0.95, spike_grad=SG, learn_beta=True, output=True)
    def forward(self, x):
        m1,m2,m3 = self.l1.init_leaky(), self.l2.init_leaky(), self.l3.init_leaky()
        sr = [[], [], []]
        for t in range(STEPS):
            s1,m1 = self.l1(self.fc1(x[t]),m1)
            s2,m2 = self.l2(self.fc2(s1),m2)
            s3,m3 = self.l3(self.fc3(s2),m3)
            sr[0].append(s1); sr[1].append(s2); sr[2].append(s3)
        return torch.stack(sr[2]).sum(0), [torch.stack(r) for r in sr], None

class HAMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.h1 = HabLayer(HIDDEN)
        self.l1 = snn.Leaky(beta=0.95, spike_grad=SG, learn_beta=True)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.h2 = HabLayer(HIDDEN)
        self.l2 = snn.Leaky(beta=0.95, spike_grad=SG, learn_beta=True)
        self.fc3 = nn.Linear(HIDDEN, 10)
        self.l3 = snn.Leaky(beta=0.95, spike_grad=SG, learn_beta=True, output=True)
    def forward(self, x):
        bs = x.shape[1]
        m1,m2,m3 = self.l1.init_leaky(), self.l2.init_leaky(), self.l3.init_leaky()
        hs1 = HabState.init(bs, HIDDEN, dev=x.device)
        hs2 = HabState.init(bs, HIDDEN, dev=x.device)
        sr = [[], [], []]
        for t in range(STEPS):
            c1,hs1 = self.h1(self.fc1(x[t]),hs1)
            s1,m1 = self.l1(c1,m1)
            c2,hs2 = self.h2(self.fc2(s1),hs2)
            s2,m2 = self.l2(c2,m2)
            s3,m3 = self.l3(self.fc3(s2),m3)
            sr[0].append(s1); sr[1].append(s2); sr[2].append(s3)
        return torch.stack(sr[2]).sum(0), [torch.stack(r) for r in sr], [hs1.strength, hs2.strength]

# Step 5: Data
def encode(imgs, steps=STEPS):
    flat = imgs.view(imgs.shape[0], -1).clamp(0, 1)
    return (torch.rand(steps, flat.shape[0], flat.shape[1], device=flat.device) < flat.unsqueeze(0)).float()

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,),(1,))])
tr_ds = datasets.FashionMNIST("/tmp/data", train=True, download=True, transform=tf)
te_ds = datasets.FashionMNIST("/tmp/data", train=False, download=True, transform=tf)
tr_dl = DataLoader(tr_ds, 128, shuffle=True, drop_last=True)
te_dl = DataLoader(te_ds, 128, shuffle=False)
print(f"Fashion-MNIST: {len(tr_ds)} train, {len(te_ds)} test", flush=True)

# Step 6: Train both
def train_model(model, is_hamt, label):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    ce = nn.CrossEntropyLoss()
    results = {"acc": [], "sr": [], "energy": []}

    for ep in range(15):
        t0 = time.time()
        model.train()
        for imgs, labels in tr_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out, recs, hab = model(encode(imgs))
            if is_hamt and hab is not None:
                # Metabolic loss
                l_task = ce(out, labels)
                ramp = min(1.0, ep / 10)
                total_s = sum(s.sum() for s in recs)
                total_n = sum(s.numel() for s in recs)
                sr_val = total_s / total_n
                l_e = (torch.relu(sr_val - 0.05)**2 + 0.1*sr_val) * ramp
                l_h = sum((h * r.mean(0)).mean() for h, r in zip(hab, recs[:len(hab)])) * ramp
                loss = l_task + 0.001 * l_e + 0.0005 * l_h
            else:
                loss = ce(out, labels)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        correct = total = 0
        srs, energies = [], []
        with torch.no_grad():
            for imgs, labels in te_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out, recs, hab = model(encode(imgs))
                correct += (out.argmax(1)==labels).sum().item()
                total += labels.size(0)
                sr = sum(s.sum().item() for s in recs) / sum(s.numel() for s in recs)
                srs.append(sr)
                fan_outs = [HIDDEN, HIDDEN, 10]
                spk_total = sum(int(s.sum().item()) for s in recs)
                syn_ops = sum(int(s.sum().item()) * fo for s, fo in zip(recs, fan_outs))
                energies.append(spk_total * 120e-12 + syn_ops * 23e-12)

        acc = correct/total
        avg_sr = np.mean(srs)
        avg_e = np.mean(energies)
        results["acc"].append(acc)
        results["sr"].append(avg_sr)
        results["energy"].append(avg_e)
        print(f"{label} Epoch {ep+1}/15: Acc={acc:.4f} SR={avg_sr:.4f} "
              f"Energy={avg_e:.2e} ({time.time()-t0:.0f}s)", flush=True)

    return results

print("\n" + "="*60, flush=True)
print("BASELINE SNN", flush=True)
print("="*60, flush=True)
torch.manual_seed(42); np.random.seed(42)
b = train_model(Baseline(), False, "Baseline")

print("\n" + "="*60, flush=True)
print("HAMT-ENHANCED SNN", flush=True)
print("="*60, flush=True)
torch.manual_seed(42); np.random.seed(42)
h = train_model(HAMT(), True, "HAMT")

# Step 7: Results
print("\n" + "="*60, flush=True)
print("RESULTS: HAMT vs Baseline on Fashion-MNIST", flush=True)
print("="*60, flush=True)
b_acc, h_acc = max(b["acc"]), max(h["acc"])
b_sr, h_sr = b["sr"][-1], h["sr"][-1]
b_e, h_e = b["energy"][-1], h["energy"][-1]
sr_red = ((b_sr - h_sr) / b_sr) * 100
e_red = ((b_e - h_e) / b_e) * 100
print(f"Best Accuracy:   Baseline={b_acc:.4f}  HAMT={h_acc:.4f}  Delta={h_acc-b_acc:+.4f}", flush=True)
print(f"Final Spike Rate: Baseline={b_sr:.4f}  HAMT={h_sr:.4f}  Reduction={sr_red:.1f}%", flush=True)
print(f"Final Energy:    Baseline={b_e:.2e}  HAMT={h_e:.2e}  Reduction={e_red:.1f}%", flush=True)
print(f"\nHAMT delivers {e_red:.0f}% energy reduction with {abs(h_acc-b_acc)*100:.1f}% accuracy difference.", flush=True)
print("Patent Pending. Contact: [your email]", flush=True)
