"""Repeat the locked MMD=0.5 comparison across random seeds.

The MMD weight was selected earlier on Batches 4-8.  This script does not tune
it again: it compares the locked MMD model (lambda=0.5) with its identical
baseline (lambda=0) on Batches 4-10 for several independent random seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from run_domain_adaptation import evaluate

ROOT = Path(__file__).parent
DEFAULT_SEEDS = [11, 23, 37]
MMD_LAMBDA = 0.5


def main(csv_path: Path, output: Path, seeds: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    rows: list[dict] = []

    for seed in seeds:
        print(f"\n{'=' * 72}\nSeed {seed}\n{'=' * 72}")
        for test_batch in range(4, 11):
            train = df[df["batch_id"] < test_batch]
            test = df[df["batch_id"] == test_batch]
            for lam, label in [(0.0, "Baseline DNN"), (MMD_LAMBDA, "MMD DNN (lambda=0.5)")]:
                row = evaluate(train, test, mmd_lambda=lam, output=output, seed=seed)
                row["model_label"] = label
                rows.append(row)
                print(
                    f"  Batch {test_batch} | {label:<21} "
                    f"Acc={row['accuracy']:.4f} F1={row['macro_f1']:.4f}"
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(output / "mmd_stability_raw.csv", index=False, encoding="utf-8-sig")

    metric_columns = [
        "accuracy", "macro_f1", "low_ppm_accuracy",
        "recall_Ethylene", "recall_Acetone", "recall_Toluene",
    ]
    summary = raw.groupby(["model_label", "test_batch"])[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "mmd_stability_by_batch.csv", index=False, encoding="utf-8-sig")

    overall = raw.groupby("model_label")[metric_columns].agg(["mean", "std"])
    overall.columns = [f"{metric}_{stat}" for metric, stat in overall.columns]
    overall = overall.reset_index()
    overall.to_csv(output / "mmd_stability_overall.csv", index=False, encoding="utf-8-sig")

    paired = raw.pivot(index=["seed", "test_batch"], columns="model_label", values="accuracy").reset_index()
    paired["mmd_minus_baseline_accuracy"] = (
        paired["MMD DNN (lambda=0.5)"] - paired["Baseline DNN"]
    )
    paired.to_csv(output / "mmd_stability_paired_accuracy.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, group in summary.groupby("model_label"):
        ax.errorbar(
            group["test_batch"], group["accuracy_mean"], yerr=group["accuracy_std"].fillna(0),
            marker="o", capsize=4, linewidth=2, label=label,
        )
    ax.set(title="MMD stability: accuracy across three random seeds", xlabel="Test Batch", ylabel="Accuracy")
    ax.set_xticks(range(4, 11))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "mmd_stability_accuracy_errorbars.png", dpi=180)
    plt.close(fig)

    print("\nSaved:", output.resolve())
    print("\nOverall mean ± std:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "mmd_stability_results")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    main(args.csv, args.output, args.seeds)
