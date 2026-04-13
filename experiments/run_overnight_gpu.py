"""
Sequential GPU benchmark with time-aware cutoff.

Runs multiple datasets on GPU with default config, stopping
before 22:30 to ensure clean shutdown.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import time
import json
from datetime import datetime
from snntorch import surrogate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.models.baseline_snn import BaselineSNN
from src.models.hamt_snn import HAMTSNN
from src.training.trainer import train_baseline, train_hamt
from src.utils.data import rate_encode
from src.utils.metrics import estimate_energy
from src.habituation.habituation_module import HabituationLayer, HabituationState
from src.losses.metabolic_loss import MetabolicLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUTOFF_HOUR = 22
CUTOFF_MINUTE = 15
RESULTS_FILE = PROJECT_ROOT / "experiments" / "results" / "local_gpu_results.json"


def time_ok():
    """Check if there's enough time before cutoff (need ~75 min per dataset)."""
    now = datetime.now()
    cutoff = now.replace(hour=CUTOFF_HOUR, minute=CUTOFF_MINUTE, second=0)
    remaining_min = (cutoff - now).total_seconds() / 60
    print(f"[CLOCK] {now.strftime('%H:%M:%S')} | {remaining_min:.0f} min until cutoff", flush=True)
    return remaining_min > 80


def get_dataset(name):
    """Load dataset, return (train_loader, test_loader, input_size, num_classes)."""
    data_dir = str(PROJECT_ROOT / "data")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])

    if name == "mnist":
        tr = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
        te = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
        return DataLoader(tr, 128, shuffle=True, drop_last=True), DataLoader(te, 128), 784, 10

    elif name == "fashion_mnist":
        tr = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
        te = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
        return DataLoader(tr, 128, shuffle=True, drop_last=True), DataLoader(te, 128), 784, 10

    elif name == "emnist_letters":
        tr = datasets.EMNIST(data_dir, split="letters", train=True, download=True, transform=tf)
        te = datasets.EMNIST(data_dir, split="letters", train=False, download=True, transform=tf)
        return DataLoader(tr, 128, shuffle=True, drop_last=True), DataLoader(te, 128), 784, 27

    elif name == "emnist_balanced":
        tr = datasets.EMNIST(data_dir, split="balanced", train=True, download=True, transform=tf)
        te = datasets.EMNIST(data_dir, split="balanced", train=False, download=True, transform=tf)
        return DataLoader(tr, 128, shuffle=True, drop_last=True), DataLoader(te, 128), 784, 47

    elif name == "emnist_digits":
        tr = datasets.EMNIST(data_dir, split="digits", train=True, download=True, transform=tf)
        te = datasets.EMNIST(data_dir, split="digits", train=False, download=True, transform=tf)
        return DataLoader(tr, 128, shuffle=True, drop_last=True), DataLoader(te, 128), 784, 10

    elif name == "conv_mnist":
        tr = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
        te = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
        return DataLoader(tr, 128, shuffle=True, drop_last=True), DataLoader(te, 128), 784, 10

    elif name == "conv_fashion":
        tr = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
        te = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
        return DataLoader(tr, 128, shuffle=True, drop_last=True), DataLoader(te, 128), 784, 10

    else:
        raise ValueError(f"Unknown dataset: {name}")


def run_test(name, use_conv=False):
    """Run baseline + HAMT comparison for one dataset."""
    print(f"\n{'='*60}", flush=True)
    print(f"DATASET: {name} ({'conv' if use_conv else 'feedforward'})", flush=True)
    print(f"{'='*60}", flush=True)

    trl, tel, inp, ncls = get_dataset(name)
    print(f"Loaded: {len(trl.dataset)} train, {len(tel.dataset)} test, {ncls} classes", flush=True)

    if use_conv:
        spike_grad = surrogate.fast_sigmoid(slope=25)

        class ConvBase(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1,16,5,padding=2); self.pool1 = nn.AvgPool2d(2)
                self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
                self.conv2 = nn.Conv2d(16,32,5,padding=2); self.pool2 = nn.AvgPool2d(2)
                self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
                self.fc = nn.Linear(32*7*7, ncls)
                self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
                self.fan_outs = [16*14*14, 32*7*7, ncls]
            def forward(self, x):
                m1,m2,m3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
                s1,s2,s3 = [],[],[]
                for t in range(25):
                    im = x[t].view(-1,1,28,28)
                    c=self.pool1(self.conv1(im)).flatten(1); sp1,m1=self.lif1(c,m1)
                    c=self.pool2(self.conv2(sp1.view(-1,16,14,14))).flatten(1); sp2,m2=self.lif2(c,m2)
                    sp3,m3=self.lif3(self.fc(sp2),m3)
                    s1.append(sp1);s2.append(sp2);s3.append(sp3)
                return torch.stack(s3).sum(0),[torch.stack(s1),torch.stack(s2),torch.stack(s3)]

        class ConvHAMT(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1,16,5,padding=2); self.pool1 = nn.AvgPool2d(2)
                self.hab1 = HabituationLayer(16*14*14); self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
                self.conv2 = nn.Conv2d(16,32,5,padding=2); self.pool2 = nn.AvgPool2d(2)
                self.hab2 = HabituationLayer(32*7*7); self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True)
                self.fc = nn.Linear(32*7*7, ncls)
                self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, learn_beta=True, output=True)
                self.fan_outs = [16*14*14, 32*7*7, ncls]
            def forward(self, x):
                bs=x.shape[1]
                m1,m2,m3=self.lif1.init_leaky(),self.lif2.init_leaky(),self.lif3.init_leaky()
                h1=HabituationState.initialize(bs,16*14*14,device=x.device)
                h2=HabituationState.initialize(bs,32*7*7,device=x.device)
                s1,s2,s3=[],[],[]
                for t in range(25):
                    im=x[t].view(-1,1,28,28)
                    c=self.pool1(self.conv1(im)).flatten(1);c,h1=self.hab1(c,h1);sp1,m1=self.lif1(c,m1)
                    c=self.pool2(self.conv2(sp1.view(-1,16,14,14))).flatten(1);c,h2=self.hab2(c,h2);sp2,m2=self.lif2(c,m2)
                    sp3,m3=self.lif3(self.fc(sp2),m3)
                    s1.append(sp1);s2.append(sp2);s3.append(sp3)
                return torch.stack(s3).sum(0),[torch.stack(s1),torch.stack(s2),torch.stack(s3)],[h1.habituation_strength,h2.habituation_strength]

        # Baseline
        print(f"\n--- Baseline Conv ---", flush=True)
        torch.manual_seed(42)
        mb = ConvBase().to(DEVICE)
        ob = torch.optim.Adam(mb.parameters(), lr=5e-4)
        lb = nn.CrossEntropyLoss()
        loss_fn_h = MetabolicLoss(lambda_energy=0.001, lambda_habituation=0.0005,
                                   target_spike_rate=0.05, ramp_epochs=10)
        b_accs, b_srs, b_es = [], [], []
        for ep in range(15):
            mb.train()
            for imgs, labels in trl:
                imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                out,recs=mb(rate_encode(imgs,25))
                loss=lb(out,labels); ob.zero_grad(); loss.backward(); ob.step()
            mb.eval(); tc=tt=0; srs=[]; es=[]
            with torch.no_grad():
                for imgs,labels in tel:
                    imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                    out,recs=mb(rate_encode(imgs,25))
                    tc+=(out.argmax(1)==labels).sum().item(); tt+=labels.size(0)
                    srs.append(sum(s.sum().item() for s in recs)/sum(s.numel() for s in recs))
                    es.append(estimate_energy(recs,mb.fan_outs))
            b_accs.append(tc/tt); b_srs.append(np.mean(srs)); b_es.append(np.mean(es))
            print(f"Base E{ep+1}: Acc={tc/tt:.4f} SR={np.mean(srs):.4f} E={np.mean(es):.2e}", flush=True)
        del mb; torch.cuda.empty_cache()

        # HAMT Conv
        print(f"\n--- HAMT Conv ---", flush=True)
        torch.manual_seed(42)
        mh = ConvHAMT().to(DEVICE)
        oh = torch.optim.Adam(mh.parameters(), lr=5e-4)
        h_accs, h_srs, h_es = [], [], []
        for ep in range(15):
            mh.train()
            for imgs,labels in trl:
                imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                out,recs,habs=mh(rate_encode(imgs,25))
                loss,_=loss_fn_h(out,labels,recs,habs,ep); oh.zero_grad(); loss.backward(); oh.step()
            mh.eval(); tc=tt=0; srs=[]; es=[]
            with torch.no_grad():
                for imgs,labels in tel:
                    imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                    out,recs,habs=mh(rate_encode(imgs,25))
                    tc+=(out.argmax(1)==labels).sum().item(); tt+=labels.size(0)
                    srs.append(sum(s.sum().item() for s in recs)/sum(s.numel() for s in recs))
                    es.append(estimate_energy(recs,mh.fan_outs))
            h_accs.append(tc/tt); h_srs.append(np.mean(srs)); h_es.append(np.mean(es))
            print(f"HAMT E{ep+1}: Acc={tc/tt:.4f} SR={np.mean(srs):.4f} E={np.mean(es):.2e}", flush=True)
        del mh; torch.cuda.empty_cache()

        ba,ha = max(b_accs),max(h_accs)
        bsr,hsr = b_srs[-1],h_srs[-1]
        be,he = b_es[-1],h_es[-1]

    else:
        # Feedforward
        print(f"\n--- Baseline FF h=800 ---", flush=True)
        torch.manual_seed(42); np.random.seed(42)
        bh = train_baseline(BaselineSNN(input_size=inp, hidden_size=800, output_size=ncls),
                           trl, tel, num_epochs=15, device=DEVICE)
        print(f"\n--- HAMT FF h=800 ---", flush=True)
        torch.manual_seed(42); np.random.seed(42)
        hh = train_hamt(HAMTSNN(input_size=inp, hidden_size=800, output_size=ncls),
                       trl, tel, num_epochs=15, lambda_energy=0.001, lambda_habituation=0.0005,
                       target_spike_rate=0.05, ramp_epochs=10, device=DEVICE)
        ba,ha = max(bh["test_acc"]),max(hh["test_acc"])
        bsr,hsr = bh["spike_rate"][-1],hh["spike_rate"][-1]
        be,he = bh["energy_joules"][-1],hh["energy_joules"][-1]

    sr_red = ((bsr - hsr) / max(bsr, 1e-10)) * 100
    e_red = ((be - he) / max(be, 1e-10)) * 100

    result = {
        "dataset": name,
        "architecture": "conv" if use_conv else "feedforward",
        "baseline_acc": round(ba, 4),
        "hamt_acc": round(ha, 4),
        "acc_delta": round(ha - ba, 4),
        "baseline_sr": round(bsr, 4),
        "hamt_sr": round(hsr, 4),
        "sr_reduction": round(sr_red, 1),
        "baseline_energy": be,
        "hamt_energy": he,
        "energy_reduction": round(e_red, 1),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }

    print(f"\nRESULT {name}:", flush=True)
    print(f"  Accuracy: B={ba:.4f} H={ha:.4f} (delta={ha-ba:+.4f})", flush=True)
    print(f"  SR: B={bsr:.4f} H={hsr:.4f} (reduction={sr_red:.1f}%)", flush=True)
    print(f"  Energy: B={be:.2e} H={he:.2e} (reduction={e_red:.1f}%)", flush=True)

    return result


def main():
    print(f"[CLOCK] Start: {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"[CLOCK] Cutoff: {CUTOFF_HOUR}:{CUTOFF_MINUTE:02d}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    # Datasets to test (in priority order)
    tests = [
        ("conv_fashion", True),     # Conv on Fashion-MNIST (new)
        ("emnist_digits", False),    # EMNIST Digits FF (new, easy dataset)
        ("conv_mnist", True),        # Conv on MNIST with default lambdas (verify local result)
    ]

    all_results = {}

    # Load existing results if any
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)

    for ds_name, use_conv in tests:
        if not time_ok():
            print(f"\n[CLOCK] Not enough time for {ds_name}. Stopping.", flush=True)
            break

        result = run_test(ds_name, use_conv)
        all_results[ds_name] = result

        # Save after each test
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Saved to {RESULTS_FILE}", flush=True)

    print(f"\n[CLOCK] Finished: {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"Total tests completed: {len(all_results)}", flush=True)


if __name__ == "__main__":
    main()
