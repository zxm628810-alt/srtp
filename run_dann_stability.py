"""Three-seed stability test: plain DNN vs DANN vs locked MMD.

DANN (lambda=0.05) is repeated for seeds 11/23/37 under the same rolling
Batch 4-10 protocol.  The already-completed MMD stability experiment supplies
the matching locked MMD(lambda=0.5) results for the same seeds, avoiding an
unnecessary second set of expensive MMD trainings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_adversarial_domain_adaptation import evaluate as evaluate_dann

ROOT = Path(__file__).parent
DEFAULT_SEEDS = [11, 23, 37]
METRICS = [
    "accuracy", "macro_f1", "low_ppm_accuracy",
    "recall_Ethylene", "recall_Acetone", "recall_Toluene",
]


def summarise(raw: pd.DataFrame, output: Path) -> None:
    by_batch = raw.groupby(["model_label", "test_batch"])[METRICS].agg(["mean", "std"])
    by_batch.columns = [f"{metric}_{stat}" for metric, stat in by_batch.columns]
    by_batch.reset_index().to_csv(output / "dann_stability_by_batch.csv", index=False, encoding="utf-8-sig")

    overall = raw.groupby("model_label")[METRICS].agg(["mean", "std"])
    overall.columns = [f"{metric}_{stat}" for metric, stat in overall.columns]
    overall.reset_index().to_csv(output / "dann_stability_overall.csv", index=False, encoding="utf-8-sig")

    batch10 = raw[raw.test_batch == 10].pivot(index="seed", columns="model_label", values="accuracy").reset_index()
    batch10["dann_minus_dnn"] = batch10["DANN (lambda=0.05)"] - batch10["Plain DNN"]
    batch10["dann_minus_mmd"] = batch10["DANN (lambda=0.05)"] - batch10["MMD (lambda=0.5)"]
    batch10.to_csv(output / "dann_stability_batch10_paired.csv", index=False, encoding="utf-8-sig")


def main(csv_path: Path, mmd_raw: Path, output: Path, seeds: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    rows: list[dict] = []

    for seed in seeds:
        print(f"\n{'=' * 64}\nDANN seed={seed}\n{'=' * 64}")
        for batch in range(4, 11):
            train, test = df[df.batch_id < batch], df[df.batch_id == batch]
            for lam, label in [(0.0, "Plain DNN"), (0.05, "DANN (lambda=0.05)")]:
                row = evaluate_dann(train, test, domain_lambda=lam, seed=seed)
                row["model_label"] = label
                rows.append(row)
                print(f"Batch {batch} | {label}: Acc={row['accuracy']:.4f}, F1={row['macro_f1']:.4f}")

    # Reuse the locked MMD lambda=0.5 measurements for exact matching seeds/batches.
    mmd = pd.read_csv(mmd_raw, encoding="utf-8-sig")
    mmd = mmd[(mmd["mmd_lambda"] == 0.5) & mmd["seed"].isin(seeds)].copy()
    required = len(seeds) * 7
    if len(mmd) != required:
        raise ValueError(f"Expected {required} MMD rows, found {len(mmd)} in {mmd_raw}")
    mmd["model_label"] = "MMD (lambda=0.5)"
    rows.extend(mmd.to_dict("records"))

    raw = pd.DataFrame(rows)
    raw.to_csv(output / "dann_stability_raw.csv", index=False, encoding="utf-8-sig")
    summarise(raw, output)
    print("\nSaved results to", output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--mmd-raw", type=Path,
                        default=ROOT / "mmd_stability_results" / "mmd_stability_raw.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "dann_stability_results")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    main(args.csv, args.mmd_raw, args.output, args.seeds)
