"""
Ablation Study: Disentangling HAMT Components

Tests 4 conditions to isolate the contribution of each component:
    1. Baseline (task loss only)
    2. Baseline + Energy loss (spike rate regularization, no habituation)
    3. Baseline + Habituation (habituation module, no energy loss)
    4. Full HAMT (habituation + energy + habituation-aware loss)

This is essential for the paper: it proves that habituation adds value
beyond simple spike rate regularization, and that the components interact
synergistically.

Run after the main comparison experiment confirms HAMT works.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import json
import time
import numpy as np

from src.models.baseline_snn import BaselineSNN
from src.models.hamt_snn import HAMTSNN
from src.training.trainer import train_baseline, train_hamt
from src.utils.data import get_static_mnist


def run_ablation(
    num_epochs: int = 20,
    batch_size: int = 128,
    num_steps: int = 25,
    hidden_size: int = 800,
    lr: float = 5e-4,
    seed: int = 42,
):
    """Run the 4-condition ablation study."""

    print("=" * 70, flush=True)
    print("ABLATION STUDY: Disentangling HAMT Components", flush=True)
    print("=" * 70, flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = PROJECT_ROOT / "experiments" / "results" / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_static_mnist(batch_size=batch_size)

    conditions = {
        "baseline": {
            "model_class": "baseline",
            "lambda_energy": 0.0,
            "lambda_habituation": 0.0,
            "description": "Task loss only (standard SNN training)",
        },
        "energy_only": {
            "model_class": "baseline",
            "lambda_energy": 0.001,
            "lambda_habituation": 0.0,
            "description": "Task + energy penalty (spike rate regularization)",
        },
        "habituation_only": {
            "model_class": "hamt",
            "lambda_energy": 0.0,
            "lambda_habituation": 0.0005,
            "description": "Task + habituation module (no energy penalty)",
        },
        "full_hamt": {
            "model_class": "hamt",
            "lambda_energy": 0.001,
            "lambda_habituation": 0.0005,
            "description": "Full HAMT (task + energy + habituation)",
        },
    }

    all_histories = {}

    for cond_name, cond_config in conditions.items():
        print(f"\n{'=' * 70}", flush=True)
        print(f"Condition: {cond_name}", flush=True)
        print(f"Description: {cond_config['description']}", flush=True)
        print(f"{'=' * 70}", flush=True)

        torch.manual_seed(seed)

        if cond_config["model_class"] == "baseline":
            model = BaselineSNN(
                input_size=784,
                hidden_size=hidden_size,
                output_size=10,
                num_steps=num_steps,
            )

            # For energy_only, use HAMT trainer with baseline model wrapped
            if cond_config["lambda_energy"] > 0:
                # Need to use the HAMT training loop but with baseline model
                # Wrap baseline to match HAMT interface
                history = _train_baseline_with_energy(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    num_epochs=num_epochs,
                    lr=lr,
                    num_steps=num_steps,
                    lambda_energy=cond_config["lambda_energy"],
                    device=device,
                )
            else:
                history = train_baseline(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    num_epochs=num_epochs,
                    lr=lr,
                    num_steps=num_steps,
                    device=device,
                )
        else:
            model = HAMTSNN(
                input_size=784,
                hidden_size=hidden_size,
                output_size=10,
                num_steps=num_steps,
            )
            history = train_hamt(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                lr=lr,
                num_steps=num_steps,
                lambda_energy=cond_config["lambda_energy"],
                lambda_habituation=cond_config["lambda_habituation"],
                device=device,
            )

        all_histories[cond_name] = history

        # Save individual history
        with open(results_dir / f"{cond_name}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(
            f"\n{cond_name} final: "
            f"Acc={history['test_acc'][-1]:.4f} "
            f"SR={history['spike_rate'][-1]:.4f} "
            f"Energy={history['energy_joules'][-1]:.2e}",
            flush=True,
        )

    # Summary comparison
    print(f"\n{'=' * 70}", flush=True)
    print("ABLATION SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n{'Condition':<25} {'Accuracy':>10} {'Spike Rate':>12} {'Energy':>12}", flush=True)
    print("-" * 60, flush=True)

    for name, hist in all_histories.items():
        print(
            f"{name:<25} "
            f"{hist['test_acc'][-1]:>10.4f} "
            f"{hist['spike_rate'][-1]:>12.4f} "
            f"{hist['energy_joules'][-1]:>12.2e}",
            flush=True,
        )

    # Save combined results
    summary = {}
    for name, hist in all_histories.items():
        summary[name] = {
            "final_accuracy": hist["test_acc"][-1],
            "best_accuracy": max(hist["test_acc"]),
            "final_spike_rate": hist["spike_rate"][-1],
            "final_energy": hist["energy_joules"][-1],
        }

    with open(results_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_dir}", flush=True)
    return all_histories


def _train_baseline_with_energy(
    model,
    train_loader,
    test_loader,
    num_epochs,
    lr,
    num_steps,
    lambda_energy,
    device,
):
    """
    Train baseline model with energy loss but no habituation.

    This isolates the effect of spike rate regularization without
    the habituation module, to show that habituation adds value
    beyond simple spike penalization.
    """
    from src.losses.metabolic_loss import MetabolicLoss
    from src.utils.data import rate_encode
    from src.utils.metrics import estimate_energy
    import torch.nn as nn

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MetabolicLoss(
        lambda_energy=lambda_energy,
        lambda_habituation=0.0,
        target_spike_rate=0.05,
        ramp_epochs=10,
    )

    history = {
        "train_loss": [], "train_acc": [], "test_acc": [],
        "spike_rate": [], "energy_joules": [], "epoch_time_sec": [],
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_correct = 0
        epoch_total = 0
        epoch_loss = 0.0
        epoch_spike_rates = []
        epoch_energy = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            spikes_in = rate_encode(images, num_steps).to(device)

            mem_out, spk_sum, spike_recs = model(spikes_in)

            loss, components = loss_fn(
                predictions=spk_sum,
                targets=labels,
                spike_recordings=spike_recs,
                habituation_strengths=None,
                current_epoch=epoch,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = spk_sum.argmax(dim=1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)
            epoch_loss += loss.item()

            with torch.no_grad():
                sr = sum(s.sum().item() for s in spike_recs) / sum(
                    s.numel() for s in spike_recs
                )
                energy = estimate_energy(spike_recs, model.get_fan_outs())
                epoch_spike_rates.append(sr)
                epoch_energy.append(energy)

            if batch_idx % 100 == 0:
                print(
                    f"  Energy-Only Epoch {epoch+1}/{num_epochs} "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss: {components['total']:.4f} SR: {sr:.4f}",
                    flush=True,
                )

        train_acc = epoch_correct / max(epoch_total, 1)

        # Test eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                spikes_in = rate_encode(images, num_steps).to(device)
                _, spk_sum, _ = model(spikes_in)
                correct += (spk_sum.argmax(1) == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / max(total, 1)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(epoch_loss / max(len(train_loader), 1))
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["spike_rate"].append(
            sum(epoch_spike_rates) / max(len(epoch_spike_rates), 1)
        )
        history["energy_joules"].append(
            sum(epoch_energy) / max(len(epoch_energy), 1)
        )
        history["epoch_time_sec"].append(epoch_time)

        print(
            f"Energy-Only Epoch {epoch+1}/{num_epochs}: "
            f"Acc={train_acc:.4f} Test={test_acc:.4f} "
            f"SR={history['spike_rate'][-1]:.4f}",
            flush=True,
        )

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HAMT ablation study")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=800)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        args.hidden = 128

    run_ablation(num_epochs=args.epochs, hidden_size=args.hidden)
