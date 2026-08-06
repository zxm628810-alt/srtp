"""Historical drift-trajectory interpolation augmentation with locked MMD.

For test Batch k, synthetic samples are generated strictly from Batches < k.
Only same-gas, same-concentration samples from adjacent historical batches are
interpolated.  This preserves the class/concentration label while exposing the
classifier to intermediate sensor-drift states.  No future-batch sample is
ever used for augmentation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from run_domain_adaptation import FEATURES, GAS_NAMES, LOW_PPM, train_model

ROOT = Path(__file__).parent
MMD_LAMBDA = 0.5
SEED = 42


def interpolate_history(train: pd.DataFrame, ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    """Create at most ratio * n historical-only midpoint drift samples."""
    target = int(len(train) * ratio)
    if target == 0:
        return train.iloc[0:0].copy()

    candidates: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
    for gas in GAS_NAMES:
        gas_data = train[train.gas_name == gas]
        for ppm, group in gas_data.groupby("concentration_ppm"):
            batches = sorted(group.batch_id.unique())
            for earlier, later in zip(batches[:-1], batches[1:]):
                left = group[group.batch_id == earlier]
                right = group[group.batch_id == later]
                pairs = min(len(left), len(right))
                if pairs:
                    candidates.append((left, right, pairs))
    if not candidates:
        return train.iloc[0:0].copy()

    weights = np.array([pair_count for _, _, pair_count in candidates], dtype=float)
    weights /= weights.sum()
    rows: list[dict] = []
    for _ in range(target):
        left, right, _ = candidates[rng.choice(len(candidates), p=weights)]
        a = left.iloc[rng.integers(len(left))]
        b = right.iloc[rng.integers(len(right))]
        # Random internal point: avoid recreating either real endpoint.
        alpha = float(rng.uniform(0.25, 0.75))
        row = {feature: (1 - alpha) * a[feature] + alpha * b[feature] for feature in FEATURES}
        row.update({
            "gas_name": a.gas_name,
            "concentration_ppm": a.concentration_ppm,
            # Treat the synthetic state as part of the latest observed history.
            "batch_id": int(b.batch_id),
            "is_synthetic": True,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(train: pd.DataFrame, test: pd.DataFrame, ratio: float, seed: int) -> dict:
    rng = np.random.default_rng(seed * 1000 + int(test.batch_id.iloc[0]) * 10 + int(ratio * 100))
    synthetic = interpolate_history(train, ratio, rng)
    augmented = pd.concat([train.assign(is_synthetic=False), synthetic], ignore_index=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(augmented[FEATURES].values)
    x_test = scaler.transform(test[FEATURES].values)
    class_map = {gas: i for i, gas in enumerate(GAS_NAMES)}
    y_train = augmented.gas_name.map(class_map).values
    model = train_model(x_train, y_train, augmented.batch_id.values, MMD_LAMBDA, seed=seed)
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32))
        pred = np.array(GAS_NAMES)[logits.argmax(1).numpy()]
    low = test.concentration_ppm.values <= LOW_PPM
    recalls = recall_score(test.gas_name, pred, labels=GAS_NAMES, average=None, zero_division=0)
    row = {
        "model": "MMD + interpolation" if ratio else "MMD baseline",
        "augmentation_ratio": ratio,
        "test_batch": int(test.batch_id.iloc[0]),
        "seed": seed,
        "n_real_train": len(train),
        "n_synthetic_train": len(synthetic),
        "accuracy": accuracy_score(test.gas_name, pred),
        "macro_f1": f1_score(test.gas_name, pred, labels=GAS_NAMES, average="macro", zero_division=0),
        "low_ppm_accuracy": accuracy_score(test.gas_name[low], pred[low]) if low.any() else np.nan,
    }
    row.update({f"recall_{gas}": float(value) for gas, value in zip(GAS_NAMES, recalls)})
    return row


def main(csv_path: Path, output: Path, ratios: list[float], seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    dev_rows = []
    for ratio in ratios:
        print(f"\nDEV augmentation ratio={ratio}")
        for batch in range(4, 9):
            row = evaluate(df[df.batch_id < batch], df[df.batch_id == batch], ratio, seed)
            dev_rows.append(row)
            print(f"  Batch {batch}: Acc={row['accuracy']:.4f}, F1={row['macro_f1']:.4f}, synth={row['n_synthetic_train']}")
    dev = pd.DataFrame(dev_rows)
    summary = dev.groupby("augmentation_ratio").agg(avg_accuracy=("accuracy", "mean"), avg_f1=("macro_f1", "mean")).reset_index()
    summary["mean_rank"] = (summary.avg_accuracy.rank(ascending=False) + summary.avg_f1.rank(ascending=False)) / 2
    summary.to_csv(output / "augmentation_dev_selection.csv", index=False, encoding="utf-8-sig")
    best_ratio = float(summary.loc[summary.mean_rank.idxmin(), "augmentation_ratio"])
    print("Selected augmentation ratio:", best_ratio)

    final_rows = []
    # If the baseline is selected, do not repeat the same final evaluation.
    final_ratios = [0.0] if best_ratio == 0.0 else [0.0, best_ratio]
    for ratio in final_ratios:
        for batch in [9, 10]:
            row = evaluate(df[df.batch_id < batch], df[df.batch_id == batch], ratio, seed)
            row["phase"] = "final"
            final_rows.append(row)
            print(f"FINAL ratio={ratio}, Batch {batch}: Acc={row['accuracy']:.4f}, F1={row['macro_f1']:.4f}")
    pd.concat([dev.assign(phase="dev"), pd.DataFrame(final_rows)], ignore_index=True).to_csv(
        output / "augmentation_results.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "drift_augmentation_results")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.csv, args.output, args.ratios, args.seed)
