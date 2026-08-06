"""Multi-seed stability check for hard-gas weighted MMD vs MMD baseline.

The single-seed experiment (seed=42) showed:
  - Batch 10 Accuracy: 71.36% → 72.11% (+0.8pp)
  - Batch 10 Ethylene Recall: 36.2% → 47.2% (+11.0pp)
  - But Acetone (-1.4pp) and Toluene (-5.2pp) decreased

This script repeats the comparison across 3 independent random seeds
(11, 23, 37) on Batches 4-10 to check whether the Batch 10 gains are
stable and whether the trade-off pattern (Ethylene up, Toluene down)
replicates consistently.

Both models use MMD lambda=0.5 (locked from earlier selection).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from run_mmd_weighted import evaluate

ROOT = Path(__file__).parent
DEFAULT_SEEDS = [11, 23, 37]


def main(csv_path: Path, output: Path, seeds: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    rows: list[dict] = []

    # ----- baseline (class_weight=1.0, low_weight=1.0, same as plain MMD) -----
    # ----- hard-gas weighted (class_weight=1.5, low_weight=1.0) -----
    variants = [
        ("MMD baseline", 1.0, 1.0),
        ("MMD hard-gas weighted", 1.5, 1.0),
    ]

    for seed in seeds:
        print(f"\n{'=' * 72}\nSeed {seed}\n{'=' * 72}")
        for test_batch in range(4, 11):
            train = df[df["batch_id"] < test_batch]
            test = df[df["batch_id"] == test_batch]
            for name, cw, lw in variants:
                row = evaluate(train, test, name, cw, lw, seed=seed)
                row["seed"] = seed
                rows.append(row)
                print(
                    f"  Batch {test_batch} | {name:<22} "
                    f"Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}  "
                    f"Ethyl={row['recall_Ethylene']:.3f}  "
                    f"Acet={row['recall_Acetone']:.3f}  "
                    f"Tol={row['recall_Toluene']:.3f}"
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(output / "mmd_weighted_stability_raw.csv", index=False, encoding="utf-8-sig")

    # ----- By-batch summary (mean +/- std across seeds) -----
    metric_cols = [
        "accuracy", "macro_f1", "low_ppm_accuracy",
        "recall_Ethylene", "recall_Acetone", "recall_Toluene",
    ]
    by_batch = raw.groupby(["variant", "test_batch"])[metric_cols].agg(["mean", "std"])
    by_batch.columns = [f"{m}_{s}" for m, s in by_batch.columns]
    by_batch = by_batch.reset_index()
    by_batch.to_csv(output / "mmd_weighted_stability_by_batch.csv", index=False, encoding="utf-8-sig")

    # ----- Overall summary -----
    overall = raw.groupby("variant")[metric_cols].agg(["mean", "std"])
    overall.columns = [f"{m}_{s}" for m, s in overall.columns]
    overall = overall.reset_index()
    overall.to_csv(output / "mmd_weighted_stability_overall.csv", index=False, encoding="utf-8-sig")

    # ----- Paired differences (per seed, per batch) -----
    paired = raw.pivot(
        index=["seed", "test_batch"], columns="variant", values="accuracy"
    ).reset_index()
    paired["weighted_minus_baseline_accuracy"] = (
        paired["MMD hard-gas weighted"] - paired["MMD baseline"]
    )
    paired.to_csv(output / "mmd_weighted_stability_paired.csv", index=False, encoding="utf-8-sig")

    # ----- Plot: accuracy error bars by batch -----
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    colours = {"MMD baseline": "#4C78A8", "MMD hard-gas weighted": "#F28E2B"}
    for label, group in by_batch.groupby("variant"):
        ax.errorbar(
            group["test_batch"], group["accuracy_mean"], yerr=group["accuracy_std"].fillna(0),
            marker="o", capsize=4, linewidth=2, color=colours[label], label=label,
        )
    ax.set(title="Weighted MMD stability: accuracy across 3 random seeds",
           xlabel="Test Batch", ylabel="Accuracy")
    ax.set_xticks(range(4, 11))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "mmd_weighted_stability_errorbars.png", dpi=200)
    plt.close(fig)

    # ----- Plot: Batch 10 per-gas Recall comparison -----
    b10 = raw[raw["test_batch"] == 10]
    gas_names = ["Ethylene", "Acetone", "Toluene"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180, sharey=True)
    for ax, gas in zip(axes, gas_names):
        col = f"recall_{gas}"
        for variant_name, colour in colours.items():
            sub = b10[b10["variant"] == variant_name]
            ax.bar(
                ["MMD\nbaseline", "MMD\nhard-gas\nweighted"],
                sub[col].values, color=colour, edgecolor="white", alpha=0.85,
            )
        ax.set_title(gas, fontsize=11)
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Batch 10 per-gas Recall: MMD baseline vs hard-gas weighted (3 seeds)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output / "mmd_weighted_stability_batch10_recall.png", dpi=200)
    plt.close(fig)

    # ----- Print summary -----
    print("\n" + "=" * 72)
    print("OVERALL (3 seeds × 7 batches)")
    print("=" * 72)
    print(overall.to_string(index=False))

    print("\n" + "=" * 72)
    print("BATCH 10 PAIRED DIFFERENCES (weighted - baseline)")
    print("=" * 72)
    b10_paired = paired[paired["test_batch"] == 10]
    print(b10_paired[["seed", "MMD baseline", "MMD hard-gas weighted", "weighted_minus_baseline_accuracy"]].to_string(index=False))
    print(f"\nMean diff: {b10_paired['weighted_minus_baseline_accuracy'].mean():+.4f}")

    print(f"\nResults saved: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "mmd_weighted_stability_results")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    main(args.csv, args.output, args.seeds)
