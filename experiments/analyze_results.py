"""
Post-experiment analysis and visualization.

Reads the comparison results and generates publication-quality figures
and a detailed numerical summary.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def load_results(results_dir: Path) -> tuple[dict, dict, dict]:
    """Load baseline history, HAMT history, and comparison summary."""
    with open(results_dir / "baseline_history.json") as f:
        baseline = json.load(f)
    with open(results_dir / "hamt_history.json") as f:
        hamt = json.load(f)
    summary_path = results_dir / "comparison_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    return baseline, hamt, summary


def plot_accuracy_vs_energy(baseline: dict, hamt: dict, save_dir: Path):
    """
    The key figure: accuracy-energy tradeoff plot.

    Each epoch is a point. HAMT should move toward the lower-left
    (lower energy, maintained accuracy) as training progresses.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    b_energy = np.array(baseline["energy_joules"])
    h_energy = np.array(hamt["energy_joules"])
    b_acc = np.array(baseline["test_acc"])
    h_acc = np.array(hamt["test_acc"])
    epochs = np.arange(1, len(b_acc) + 1)

    # Plot trajectories with epoch color gradient
    scatter_b = ax.scatter(
        b_energy, b_acc, c=epochs, cmap="Blues", s=50,
        edgecolors="blue", linewidths=0.5, label="Baseline", zorder=3,
    )
    scatter_h = ax.scatter(
        h_energy, h_acc, c=epochs, cmap="Reds", s=50,
        edgecolors="red", linewidths=0.5, label="HAMT", zorder=3,
    )

    # Connect points with lines
    ax.plot(b_energy, b_acc, "b-", alpha=0.3, linewidth=1)
    ax.plot(h_energy, h_acc, "r-", alpha=0.3, linewidth=1)

    # Mark final epochs
    ax.scatter([b_energy[-1]], [b_acc[-1]], c="blue", s=150, marker="*",
               zorder=4, edgecolors="black", linewidths=1)
    ax.scatter([h_energy[-1]], [h_acc[-1]], c="red", s=150, marker="*",
               zorder=4, edgecolors="black", linewidths=1)

    ax.set_xlabel("Estimated Energy per Batch (Joules)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Energy Tradeoff: HAMT vs Baseline")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter_h, ax=ax, label="Epoch")
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_vs_energy.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_dir / 'accuracy_vs_energy.png'}", flush=True)


def plot_habituation_dynamics(hamt: dict, save_dir: Path):
    """
    Show how habituation parameters evolve during training.

    This reveals what the network learned about when to suppress.
    """
    if "habituation_stats" not in hamt or not hamt["habituation_stats"]:
        print("No habituation stats available", flush=True)
        return

    stats = hamt["habituation_stats"]
    epochs = range(1, len(stats) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Alpha (suppression strength)
    ax = axes[0]
    ax.plot(epochs, [s["layer1_alpha_mean"] for s in stats], "b-o",
            markersize=3, label="Layer 1")
    ax.plot(epochs, [s["layer2_alpha_mean"] for s in stats], "r-s",
            markersize=3, label="Layer 2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Alpha (suppression strength)")
    ax.set_title("Learned Suppression Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tau hab (habituation rate)
    ax = axes[1]
    ax.plot(epochs, [s["layer1_tau_hab_mean"] for s in stats], "b-o",
            markersize=3, label="Layer 1")
    ax.plot(epochs, [s["layer2_tau_hab_mean"] for s in stats], "r-s",
            markersize=3, label="Layer 2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tau habituation")
    ax.set_title("Learned Habituation Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tau rec (recovery rate)
    ax = axes[2]
    ax.plot(epochs, [s["layer1_tau_rec_mean"] for s in stats], "b-o",
            markersize=3, label="Layer 1")
    ax.plot(epochs, [s["layer2_tau_rec_mean"] for s in stats], "r-s",
            markersize=3, label="Layer 2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tau recovery")
    ax.set_title("Learned Recovery Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Habituation Parameter Evolution During Training", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "habituation_dynamics.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_dir / 'habituation_dynamics.png'}", flush=True)


def plot_spike_rate_over_time(baseline: dict, hamt: dict, save_dir: Path):
    """
    Show spike rate reduction over training epochs.

    The gap between baseline and HAMT should widen as training progresses
    (progressive efficiency from habituation).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(baseline["spike_rate"]) + 1)

    ax.plot(epochs, baseline["spike_rate"], "b-o", markersize=4, label="Baseline SNN",
            linewidth=2)
    ax.plot(epochs, hamt["spike_rate"], "r-s", markersize=4, label="HAMT-SNN",
            linewidth=2)
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7,
               label="Cortical target (5%)", linewidth=1.5)

    # Shade the efficiency gap
    b_sr = np.array(baseline["spike_rate"])
    h_sr = np.array(hamt["spike_rate"])
    ep = np.array(list(epochs))
    ax.fill_between(ep, h_sr, b_sr, alpha=0.15, color="green",
                    where=b_sr > h_sr, label="Energy savings")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Spike Rate")
    ax.set_title("Progressive Spike Rate Reduction: HAMT vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "spike_rate_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_dir / 'spike_rate_comparison.png'}", flush=True)


def print_summary(baseline: dict, hamt: dict):
    """Print numerical comparison summary."""
    print("\n" + "=" * 60, flush=True)
    print("NUMERICAL SUMMARY", flush=True)
    print("=" * 60, flush=True)

    metrics = [
        ("Test Accuracy (final)", baseline["test_acc"][-1], hamt["test_acc"][-1]),
        ("Test Accuracy (best)", max(baseline["test_acc"]), max(hamt["test_acc"])),
        ("Spike Rate (final)", baseline["spike_rate"][-1], hamt["spike_rate"][-1]),
        ("Energy J/batch (final)", baseline["energy_joules"][-1], hamt["energy_joules"][-1]),
    ]

    print(f"\n{'Metric':<30} {'Baseline':>12} {'HAMT':>12} {'Delta':>12}", flush=True)
    print("-" * 66, flush=True)

    for name, b_val, h_val in metrics:
        if "Accuracy" in name:
            delta = f"{(h_val - b_val):+.4f}"
            print(f"{name:<30} {b_val:>12.4f} {h_val:>12.4f} {delta:>12}", flush=True)
        elif "Rate" in name:
            reduction = ((b_val - h_val) / max(b_val, 1e-8)) * 100
            print(f"{name:<30} {b_val:>12.4f} {h_val:>12.4f} {reduction:>+11.1f}%", flush=True)
        else:
            reduction = ((b_val - h_val) / max(b_val, 1e-8)) * 100
            print(f"{name:<30} {b_val:>12.2e} {h_val:>12.2e} {reduction:>+11.1f}%", flush=True)

    # Efficiency ratio
    b_eff = max(baseline["test_acc"]) / max(max(baseline["energy_joules"]), 1e-15)
    h_eff = max(hamt["test_acc"]) / max(max(hamt["energy_joules"]), 1e-15)
    print(f"\n{'Accuracy/Energy Ratio':<30} {b_eff:>12.2f} {h_eff:>12.2f} "
          f"{((h_eff - b_eff) / max(b_eff, 1e-8)) * 100:>+11.1f}%", flush=True)


if __name__ == "__main__":
    results_dir = PROJECT_ROOT / "experiments" / "results" / "mnist_comparison"

    if not (results_dir / "baseline_history.json").exists():
        print("No results found. Run run_mnist_comparison.py first.", flush=True)
        sys.exit(1)

    baseline, hamt, summary = load_results(results_dir)

    print_summary(baseline, hamt)
    plot_accuracy_vs_energy(baseline, hamt, results_dir)
    plot_spike_rate_over_time(baseline, hamt, results_dir)
    plot_habituation_dynamics(hamt, results_dir)

    print("\nAll analysis complete.", flush=True)
