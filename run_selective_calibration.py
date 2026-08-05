"""Per-sensor selective calibration: only calibrate sensors that show significant drift.

Key idea: the per-feature KS analysis showed that not all 128 sensors drift equally.
Batch 7 had only 38 features with KS >= 0.2, while Batch 8 had 124.  If we blindly
apply historical baseline calibration to all 128 sensors, we may be adding noise to
stable sensors that don't need it.

This experiment adds a new model kind that:
  1. Looks at TRAINING data only (batches < k)
  2. Computes KS statistic for each feature between the earliest and latest training batch
  3. Only computes history_relative calibration features for sensors with KS >= 0.2
  4. Stable sensors (KS < 0.2) keep their raw values without calibration

All decisions are made using training data only - no test information leakage.

The baseline for comparison is dnn_sensor_history_baseline (all 128 sensors calibrated).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
FEATURES = [f"feature_{i}" for i in range(1, 129)]
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
LOW_CONCENTRATION_PPM = 50.0
KS_THRESHOLD = 0.20


def make_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu",
                               early_stopping=True, validation_fraction=.15,
                               max_iter=300, random_state=42))
    ])


def identify_drift_prone_features(train: pd.DataFrame, threshold: float = KS_THRESHOLD) -> list[str]:
    """Identify which sensors drift significantly within the training data.

    Compares the earliest and latest training batches using per-feature KS test.
    Only training data is used - no test batch information.
    Returns list of feature names where KS >= threshold.
    """
    batches = sorted(train.batch_id.unique())
    if len(batches) < 2:
        return list(FEATURES)

    earliest = train[train.batch_id == batches[0]][FEATURES]
    latest = train[train.batch_id == batches[-1]][FEATURES]

    drift_features = []
    for feat in FEATURES:
        ks = ks_2samp(earliest[feat].values, latest[feat].values).statistic
        if ks >= threshold:
            drift_features.append(feat)
    return drift_features


def sliding_history_relative_selective(
    frame: pd.DataFrame, history: pd.DataFrame, window: int, selected_features: list[str]
) -> pd.DataFrame:
    """Same logic as sliding_history_relative, but only for selected features."""
    if window < 1:
        raise ValueError("history window must be at least 1")
    if not selected_features:
        return pd.DataFrame(index=frame.index)

    fallback = history.loc[history.batch_id == history.batch_id.min(), selected_features].median()
    relative_parts = []
    for batch in sorted(frame.batch_id.unique()):
        available = sorted(history.loc[history.batch_id < batch, "batch_id"].unique())
        chosen = available[-window:]
        baseline = history.loc[history.batch_id.isin(chosen), selected_features].median() if chosen else fallback
        safe_baseline = baseline.replace(0, 1e-8)
        current = frame.loc[frame.batch_id == batch, selected_features]
        relative = current.divide(safe_baseline, axis="columns").subtract(1.0)
        relative_parts.append(relative)

    result = pd.concat(relative_parts).reindex(frame.index)
    result.columns = [f"history_relative_{name}" for name in selected_features]
    return result


def build_features(frame: pd.DataFrame, history: pd.DataFrame, selective: bool,
                   history_window: int = 3) -> tuple[pd.DataFrame, int, int]:
    """Build feature matrix. Returns (X, n_calibrated, n_stable)."""
    x = frame[FEATURES].copy()

    if selective:
        drift_features = identify_drift_prone_features(history)
    else:
        drift_features = list(FEATURES)

    n_calibrated = len(drift_features)
    n_stable = 128 - n_calibrated

    if drift_features:
        relative = sliding_history_relative_selective(frame, history, history_window, drift_features)
        x = pd.concat([x, relative], axis=1)

    return x, n_calibrated, n_stable


def evaluate_batch(train: pd.DataFrame, test: pd.DataFrame, output: Path, tag: str,
                   history_window: int, selective: bool) -> dict:
    """Train and evaluate one batch."""

    x_train, n_cal, n_stable = build_features(train, train, selective, history_window)
    x_test, _, _ = build_features(test, train, selective, history_window)

    model = make_model()
    model.fit(x_train, train["gas_name"])
    pred = model.predict(x_test)

    test_batch = int(test.batch_id.iloc[0])
    row = {
        "experiment": tag,
        "model": "dnn_selective" if selective else "dnn_all_calibrated",
        "test_batch": test_batch,
        "n_train": len(train),
        "n_test": len(test),
        "n_calibrated_features": n_cal,
        "n_stable_features": n_stable,
        "calibrated_pct": round(n_cal / 128 * 100, 1),
        "accuracy": accuracy_score(test.gas_name, pred),
        "macro_f1": f1_score(test.gas_name, pred, labels=GAS_NAMES, average="macro", zero_division=0),
    }

    recalls = recall_score(test.gas_name, pred, labels=GAS_NAMES, average=None, zero_division=0)
    row.update({f"recall_{gas}": float(v) for gas, v in zip(GAS_NAMES, recalls)})

    low_mask = test.concentration_ppm <= LOW_CONCENTRATION_PPM
    row["low_ppm_n"] = int(low_mask.sum())
    row["low_ppm_accuracy"] = accuracy_score(
        test.loc[low_mask, "gas_name"], pred[low_mask]
    ) if low_mask.any() else np.nan

    cm = pd.DataFrame(
        confusion_matrix(test.gas_name, pred, labels=GAS_NAMES),
        index=GAS_NAMES, columns=GAS_NAMES,
    )
    suffix = "selective" if selective else "all"
    cm.to_csv(output / f"confusion_{suffix}_batch{test_batch}.csv", encoding="utf-8-sig")

    return row


def main(csv_path: Path, output: Path, test_batches: list[int], history_window: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    all_rows = []

    # ===== BASELINE: calibrate ALL 128 sensors =====
    print("=" * 70)
    print("BASELINE: All 128 sensors calibrated")
    print("=" * 70)
    for test_batch in test_batches:
        train = df[df["batch_id"] < test_batch]
        test = df[df["batch_id"] == test_batch]
        row = evaluate_batch(train, test, output, "rolling", history_window, selective=False)
        all_rows.append(row)
        print(f"  Batch {test_batch}: Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}")

    # ===== SELECTIVE: only calibrate drift-prone sensors =====
    print(f"\n{'=' * 70}")
    print(f"SELECTIVE: Only calibrate sensors with KS >= {KS_THRESHOLD}")
    print("=" * 70)
    for test_batch in test_batches:
        train = df[df["batch_id"] < test_batch]
        test = df[df["batch_id"] == test_batch]
        row = evaluate_batch(train, test, output, "rolling", history_window, selective=True)
        all_rows.append(row)
        print(f"  Batch {test_batch}: {row['n_calibrated_features']}/128 sensors calibrated "
              f"({row['calibrated_pct']}%)  Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}")

    # ===== SAVE =====
    results = pd.DataFrame(all_rows)
    results.to_csv(output / "selective_calibration_results.csv", index=False, encoding="utf-8-sig")

    # ===== PRINT COMPARISON =====
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Batch':<7} {'All-Cal':<12} {'Selective':<14} {'Delta':<8} {'Calibrated%':<12}")
    print("-" * 57)
    for batch in test_batches:
        all_row = results[(results["test_batch"] == batch) & (results["model"] == "dnn_all_calibrated")]
        sel_row = results[(results["test_batch"] == batch) & (results["model"] == "dnn_selective")]
        if not all_row.empty and not sel_row.empty:
            all_acc = all_row.iloc[0]["accuracy"]
            sel_acc = sel_row.iloc[0]["accuracy"]
            cal_pct = sel_row.iloc[0]["calibrated_pct"]
            delta = sel_acc - all_acc
            sign = "+" if delta > 0 else ""
            print(f"Batch {batch:<2}  {all_acc:<12.4f} {sel_acc:<14.4f} {sign}{delta:.4f}    {cal_pct}%")

    avg_all = results[results["model"] == "dnn_all_calibrated"]["accuracy"].mean()
    avg_sel = results[results["model"] == "dnn_selective"]["accuracy"].mean()
    print(f"\n{'Avg':<7} {avg_all:<12.4f} {avg_sel:<14.4f} {'+' if avg_sel > avg_all else ''}{avg_sel - avg_all:.4f}")

    print(f"\nResults: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "selective_calibration_results")
    parser.add_argument("--test-batches", nargs="+", type=int,
                        choices=range(4, 11), default=list(range(4, 11)))
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument("--ks-threshold", type=float, default=KS_THRESHOLD)
    args = parser.parse_args()
    main(args.csv, args.output, args.test_batches, args.history_window)
