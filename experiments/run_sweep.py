"""
Hyperparameter Sweep Runner.

Runs a grid search over key HAMT hyperparameters to find the
configuration that maximizes energy reduction while maintaining accuracy.
Each configuration runs for fewer epochs (to save time) and the best
configurations are then validated with full training.

Usage:
    python run_sweep.py --param lambda_energy --quick
    python run_sweep.py --param hidden_size
    python run_sweep.py --param aggressive_hamt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import json
import numpy as np
import time
import argparse

from src.models.baseline_snn import BaselineSNN
from src.models.hamt_snn import HAMTSNN
from src.training.trainer import train_baseline, train_hamt
from src.utils.data import get_static_mnist


def run_single_config(
    config: dict,
    train_loader,
    test_loader,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Run a single HAMT configuration and return metrics."""
    torch.manual_seed(seed)

    model = HAMTSNN(
        input_size=784,
        hidden_size=config.get("hidden", 800),
        output_size=10,
        num_steps=config.get("num_steps", 25),
        hab_alpha=config.get("hab_alpha", 0.5),
        hab_tau=config.get("hab_tau", 0.9),
        hab_recovery=config.get("hab_recovery", 0.8),
        history_length=config.get("history_length", 5),
    )

    history = train_hamt(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config.get("epochs", 15),
        lr=config.get("lr", 5e-4),
        num_steps=config.get("num_steps", 25),
        lambda_energy=config.get("lambda_energy", 0.001),
        lambda_habituation=config.get("lambda_habituation", 0.0005),
        target_spike_rate=config.get("target_spike_rate", 0.05),
        device=device,
    )

    return {
        "config": config,
        "best_acc": max(history["test_acc"]),
        "final_acc": history["test_acc"][-1],
        "final_spike_rate": history["spike_rate"][-1],
        "final_energy": history["energy_joules"][-1],
        "habituation_stats": history["habituation_stats"][-1] if history.get("habituation_stats") else {},
        "accuracy_per_energy": max(history["test_acc"]) / max(history["energy_joules"][-1], 1e-20),
    }


def run_sweep(param_name: str, quick: bool = False):
    """Run sweep over a single parameter."""

    print(f"{'=' * 70}", flush=True)
    print(f"HYPERPARAMETER SWEEP: {param_name}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Load sweep config
    with open(PROJECT_ROOT / "experiments" / "sweep_configs.json") as f:
        all_configs = json.load(f)

    if param_name not in all_configs["sweeps"]:
        print(f"Unknown parameter: {param_name}", flush=True)
        print(f"Available: {list(all_configs['sweeps'].keys())}", flush=True)
        return

    sweep = all_configs["sweeps"][param_name]
    values = sweep["values"]
    fixed = sweep["fixed"].copy()

    if quick:
        fixed["epochs"] = min(fixed.get("epochs", 15), 5)
        if "hidden" not in fixed:
            fixed["hidden"] = 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Purpose: {sweep['purpose']}", flush=True)
    print(f"Values to test: {values}", flush=True)
    print(f"Fixed params: {fixed}", flush=True)

    # Load data once
    batch_size = fixed.get("batch_size", 128)
    train_loader, test_loader = get_static_mnist(batch_size=batch_size)

    # Results directory
    results_dir = PROJECT_ROOT / "experiments" / "results" / f"sweep_{param_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run baseline once for comparison
    print(f"\nRunning baseline for comparison...", flush=True)
    torch.manual_seed(42)
    baseline = BaselineSNN(
        input_size=784,
        hidden_size=fixed.get("hidden", 800),
        output_size=10,
        num_steps=fixed.get("num_steps", 25),
    )
    baseline_history = train_baseline(
        model=baseline,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=fixed.get("epochs", 15),
        lr=fixed.get("lr", 5e-4),
        num_steps=fixed.get("num_steps", 25),
        device=device,
    )
    baseline_result = {
        "best_acc": max(baseline_history["test_acc"]),
        "final_spike_rate": baseline_history["spike_rate"][-1],
        "final_energy": baseline_history["energy_joules"][-1],
    }

    # Run each value
    results = []
    for i, val in enumerate(values):
        print(f"\n{'=' * 50}", flush=True)
        print(f"Config {i+1}/{len(values)}: {param_name}={val}", flush=True)
        print(f"{'=' * 50}", flush=True)

        config = fixed.copy()
        if isinstance(val, dict):
            config.update(val)
        else:
            config[param_name] = val

        start = time.time()
        result = run_single_config(config, train_loader, test_loader, device)
        result["wall_time_sec"] = time.time() - start

        # Compare to baseline
        result["acc_vs_baseline"] = result["best_acc"] - baseline_result["best_acc"]
        result["sr_reduction_pct"] = (
            (baseline_result["final_spike_rate"] - result["final_spike_rate"])
            / max(baseline_result["final_spike_rate"], 1e-10)
        ) * 100
        result["energy_reduction_pct"] = (
            (baseline_result["final_energy"] - result["final_energy"])
            / max(baseline_result["final_energy"], 1e-10)
        ) * 100

        results.append(result)

        print(
            f"\nResult: Acc={result['best_acc']:.4f} "
            f"(vs baseline: {result['acc_vs_baseline']:+.4f}) "
            f"SR reduction: {result['sr_reduction_pct']:.1f}% "
            f"Energy reduction: {result['energy_reduction_pct']:.1f}%",
            flush=True,
        )

    # Summary table
    print(f"\n{'=' * 70}", flush=True)
    print(f"SWEEP SUMMARY: {param_name}", flush=True)
    print(f"Baseline: Acc={baseline_result['best_acc']:.4f} "
          f"SR={baseline_result['final_spike_rate']:.4f} "
          f"Energy={baseline_result['final_energy']:.2e}", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n{'Value':<15} {'Accuracy':>10} {'Delta Acc':>10} "
          f"{'SR Red%':>10} {'Energy Red%':>12}", flush=True)
    print("-" * 58, flush=True)

    best_idx = 0
    best_score = -999

    for i, r in enumerate(results):
        val_str = str(values[i])[:14]
        print(
            f"{val_str:<15} {r['best_acc']:>10.4f} "
            f"{r['acc_vs_baseline']:>+10.4f} "
            f"{r['sr_reduction_pct']:>+10.1f} "
            f"{r['energy_reduction_pct']:>+12.1f}",
            flush=True,
        )

        # Score: maximize energy reduction while keeping accuracy within 2%
        score = r["energy_reduction_pct"]
        if r["acc_vs_baseline"] < -0.02:
            score -= 100  # Heavy penalty for >2% accuracy loss
        if score > best_score:
            best_score = score
            best_idx = i

    print(f"\nBest config: {param_name}={values[best_idx]} "
          f"(score={best_score:.1f})", flush=True)

    # Save all results
    output = {
        "param_name": param_name,
        "baseline": baseline_result,
        "results": results,
        "best_index": best_idx,
        "best_value": values[best_idx],
    }
    with open(results_dir / "sweep_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to: {results_dir / 'sweep_results.json'}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAMT hyperparameter sweep")
    parser.add_argument("--param", type=str, required=True,
                        help="Parameter to sweep (see sweep_configs.json)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer epochs, smaller model)")
    args = parser.parse_args()

    run_sweep(args.param, quick=args.quick)
