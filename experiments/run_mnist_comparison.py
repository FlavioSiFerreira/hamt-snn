"""
Experiment 1: HAMT vs Baseline SNN on MNIST

Runs both models on rate-coded MNIST with identical architectures,
hyperparameters, and random seeds. Compares:
    1. Classification accuracy (HAMT should match or be within 1-2%)
    2. Spike rate (HAMT should be significantly lower)
    3. Estimated energy consumption (HAMT should be 30-50% lower)
    4. Habituation dynamics (how learned parameters evolve)

This is the minimum viable experiment to validate the HAMT concept.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import json
import time
import matplotlib.pyplot as plt
import numpy as np

from src.models.baseline_snn import BaselineSNN
from src.models.hamt_snn import HAMTSNN
from src.training.trainer import train_baseline, train_hamt
from src.utils.data import get_static_mnist


def run_comparison(
    num_epochs: int = 20,
    batch_size: int = 128,
    num_steps: int = 25,
    hidden_size: int = 800,
    lr: float = 5e-4,
    seed: int = 42,
    lambda_energy: float = 0.001,
    lambda_habituation: float = 0.0005,
    target_spike_rate: float = 0.05,
):
    """Run full comparison experiment."""

    print("=" * 70, flush=True)
    print("HAMT vs Baseline SNN Comparison on MNIST", flush=True)
    print("=" * 70, flush=True)

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Results directory
    results_dir = PROJECT_ROOT / "experiments" / "results" / "mnist_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading MNIST...", flush=True)
    train_loader, test_loader = get_static_mnist(
        batch_size=batch_size,
        num_steps=num_steps,
    )
    print(
        f"Train: {len(train_loader.dataset)} samples, "
        f"Test: {len(test_loader.dataset)} samples",
        flush=True,
    )

    # Save experiment config
    config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "hidden_size": hidden_size,
        "lr": lr,
        "seed": seed,
        "lambda_energy": lambda_energy,
        "lambda_habituation": lambda_habituation,
        "target_spike_rate": target_spike_rate,
        "device": str(device),
    }
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- BASELINE ----
    print("\n" + "=" * 70, flush=True)
    print("PHASE 1: Training Baseline SNN (no energy awareness)", flush=True)
    print("=" * 70, flush=True)

    torch.manual_seed(seed)
    baseline_model = BaselineSNN(
        input_size=784,
        hidden_size=hidden_size,
        output_size=10,
        num_steps=num_steps,
    )
    print(
        f"Baseline params: {sum(p.numel() for p in baseline_model.parameters()):,}",
        flush=True,
    )

    baseline_history = train_baseline(
        model=baseline_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        lr=lr,
        num_steps=num_steps,
        device=device,
        save_dir=results_dir,
    )

    # ---- HAMT ----
    print("\n" + "=" * 70, flush=True)
    print("PHASE 2: Training HAMT-SNN (habituation + metabolic cost)", flush=True)
    print("=" * 70, flush=True)

    torch.manual_seed(seed)
    hamt_model = HAMTSNN(
        input_size=784,
        hidden_size=hidden_size,
        output_size=10,
        num_steps=num_steps,
    )
    print(
        f"HAMT params: {sum(p.numel() for p in hamt_model.parameters()):,} "
        f"(+{sum(p.numel() for p in hamt_model.parameters()) - sum(p.numel() for p in baseline_model.parameters()):,} "
        f"from habituation)",
        flush=True,
    )

    hamt_history = train_hamt(
        model=hamt_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        lr=lr,
        num_steps=num_steps,
        lambda_energy=lambda_energy,
        lambda_habituation=lambda_habituation,
        target_spike_rate=target_spike_rate,
        device=device,
        save_dir=results_dir,
    )

    # ---- COMPARISON ----
    print("\n" + "=" * 70, flush=True)
    print("RESULTS COMPARISON", flush=True)
    print("=" * 70, flush=True)

    b_acc = baseline_history["test_acc"][-1]
    h_acc = hamt_history["test_acc"][-1]
    b_sr = baseline_history["spike_rate"][-1]
    h_sr = hamt_history["spike_rate"][-1]
    b_energy = baseline_history["energy_joules"][-1]
    h_energy = hamt_history["energy_joules"][-1]

    print(f"\nTest Accuracy:     Baseline={b_acc:.4f}  HAMT={h_acc:.4f}  "
          f"Delta={h_acc - b_acc:+.4f}", flush=True)
    print(f"Spike Rate:        Baseline={b_sr:.4f}  HAMT={h_sr:.4f}  "
          f"Reduction={((b_sr - h_sr) / max(b_sr, 1e-8)) * 100:.1f}%", flush=True)
    print(f"Energy (J/batch):  Baseline={b_energy:.2e}  HAMT={h_energy:.2e}  "
          f"Reduction={((b_energy - h_energy) / max(b_energy, 1e-8)) * 100:.1f}%",
          flush=True)

    # Habituation analysis
    hab_stats = hamt_history["habituation_stats"][-1]
    print(f"\nLearned Habituation Parameters:", flush=True)
    for key, val in hab_stats.items():
        label = key.replace("_", " ").title()
        print(f"  {label}: {val:.4f}", flush=True)

    # Generate comparison plots
    generate_plots(baseline_history, hamt_history, results_dir)

    # Save summary
    summary = {
        "baseline_final_acc": b_acc,
        "hamt_final_acc": h_acc,
        "accuracy_delta": h_acc - b_acc,
        "baseline_final_spike_rate": b_sr,
        "hamt_final_spike_rate": h_sr,
        "spike_rate_reduction_pct": ((b_sr - h_sr) / max(b_sr, 1e-8)) * 100,
        "baseline_final_energy": b_energy,
        "hamt_final_energy": h_energy,
        "energy_reduction_pct": ((b_energy - h_energy) / max(b_energy, 1e-8)) * 100,
        "habituation_params": hab_stats,
    }
    with open(results_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_dir}", flush=True)
    return summary


def generate_plots(
    baseline: dict,
    hamt: dict,
    save_dir: Path,
):
    """Generate comparison visualization plots."""

    epochs = range(1, len(baseline["test_acc"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("HAMT vs Baseline SNN on MNIST", fontsize=14, fontweight="bold")

    # Plot 1: Test Accuracy
    ax = axes[0, 0]
    ax.plot(epochs, baseline["test_acc"], "b-o", label="Baseline", markersize=3)
    ax.plot(epochs, hamt["test_acc"], "r-s", label="HAMT", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Classification Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Spike Rate
    ax = axes[0, 1]
    ax.plot(epochs, baseline["spike_rate"], "b-o", label="Baseline", markersize=3)
    ax.plot(epochs, hamt["spike_rate"], "r-s", label="HAMT", markersize=3)
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.5, label="Cortical target (5%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Spike Rate")
    ax.set_title("Spike Rate (lower = more efficient)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Energy
    ax = axes[1, 0]
    ax.plot(epochs, baseline["energy_joules"], "b-o", label="Baseline", markersize=3)
    ax.plot(epochs, hamt["energy_joules"], "r-s", label="HAMT", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Estimated Energy (Joules/batch)")
    ax.set_title("Energy Consumption")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Plot 4: HAMT loss components
    ax = axes[1, 1]
    if "train_loss_task" in hamt:
        ax.plot(epochs, hamt["train_loss_task"], "g-", label="Task Loss", linewidth=2)
        ax.plot(epochs, hamt["train_loss_energy"], "orange", label="Energy Loss", linewidth=2)
        ax.plot(epochs, hamt["train_loss_habituation"], "purple", label="Habituation Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Component")
        ax.set_title("HAMT Loss Decomposition")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "comparison_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to: {save_dir / 'comparison_plots.png'}", flush=True)


if __name__ == "__main__":
    # Quick test run (fewer epochs for validation)
    import argparse

    parser = argparse.ArgumentParser(description="HAMT vs Baseline SNN comparison")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--steps", type=int, default=25, help="SNN timesteps")
    parser.add_argument("--hidden", type=int, default=800, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick test (3 epochs, small hidden)")

    args = parser.parse_args()

    if args.quick:
        args.epochs = 3
        args.hidden = 128
        print("QUICK TEST MODE: 3 epochs, hidden=128", flush=True)

    run_comparison(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_steps=args.steps,
        hidden_size=args.hidden,
        lr=args.lr,
    )
