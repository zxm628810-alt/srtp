"""Strict temporal drift experiments for the UCI gas-sensor dataset.

Models:
  rf_sensor        RandomForest baseline using only 128 sensor features
  dnn_sensor       MLP neural network using only sensor features
  dnn_sensor_time  MLP using sensor features plus known batch/time feature
  dnn_sensor_baseline  MLP using raw features plus changes relative to the
                       earliest available sensor baseline
  dnn_sensor_history_baseline  MLP using raw features plus a sliding baseline
                               calculated only from earlier historical batches
  dnn_sensor_weighted  MLP that upweights difficult gases and low-concentration
                       training samples

The test batches are never included in model fitting or scaler fitting.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [f"feature_{i}" for i in range(1, 129)]
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
DIFFICULT_GASES = {"Ethylene", "Acetone", "Toluene"}
LOW_CONCENTRATION_PPM = 50.0

def make_model(kind: str):
    if kind == "rf_sensor":
        return RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu",
                               early_stopping=True, validation_fraction=.15,
                               max_iter=300, random_state=42))
    ])

def sliding_history_relative(frame: pd.DataFrame, history: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return per-sample changes against a baseline from earlier batches only.

    For Batch b, the baseline is the feature-wise median of the latest
    ``window`` batches strictly earlier than b.  For the first available
    training batch, its own median is used only as a fallback so feature values
    remain finite.  Test-batch rows never contribute to any baseline.
    """
    if window < 1:
        raise ValueError("history window must be at least 1")
    fallback = history.loc[history.batch_id == history.batch_id.min(), FEATURES].median()
    relative_parts = []
    for batch in sorted(frame.batch_id.unique()):
        available = sorted(history.loc[history.batch_id < batch, "batch_id"].unique())
        chosen = available[-window:]
        baseline = history.loc[history.batch_id.isin(chosen), FEATURES].median() if chosen else fallback
        safe_baseline = baseline.replace(0, 1e-8)
        current = frame.loc[frame.batch_id == batch, FEATURES]
        relative = current.divide(safe_baseline, axis="columns").subtract(1.0)
        relative_parts.append(relative)
    result = pd.concat(relative_parts).reindex(frame.index)
    result.columns = [f"history_relative_{name}" for name in FEATURES]
    return result


def select_x(frame: pd.DataFrame, kind: str, baseline: pd.Series | None = None,
             history: pd.DataFrame | None = None, history_window: int = 3) -> pd.DataFrame:
    x = frame[FEATURES].copy()
    if kind == "dnn_sensor_time":
        # Device age/batch is assumed known at inference. It is not fitted using test labels.
        x["time_batch"] = frame["batch_id"].astype(float)
    elif kind == "dnn_sensor_baseline":
        if baseline is None:
            raise ValueError("基线相对特征需要训练集传感器基线")
        # Baseline comes only from the earliest batch in the training history.
        # Thus future test-batch values or labels never influence this transform.
        safe_baseline = baseline.replace(0, 1e-8)
        relative = x.divide(safe_baseline, axis="columns").subtract(1.0)
        relative.columns = [f"relative_{name}" for name in FEATURES]
        x = pd.concat([x, relative], axis=1)
    elif kind == "dnn_sensor_history_baseline":
        if history is None:
            raise ValueError("sliding history features require historical training batches")
        x = pd.concat([x, sliding_history_relative(frame, history, history_window)], axis=1)
    return x


def difficult_sample_weights(train: pd.DataFrame) -> np.ndarray:
    """Training-only weights for the error-prone gases and low-ppm samples.

    The rules are fixed before evaluation: Ethylene, Acetone and Toluene get a
    factor of 2; samples at or below 50 ppm get a further factor of 1.5.  Test
    labels are never used for fitting or for choosing these weights.
    """
    weights = np.ones(len(train), dtype=float)
    weights[train.gas_name.isin(DIFFICULT_GASES).to_numpy()] *= 2.0
    weights[(train.concentration_ppm <= LOW_CONCENTRATION_PPM).to_numpy()] *= 1.5
    return weights

def evaluate(train: pd.DataFrame, test: pd.DataFrame, kind: str, output: Path, tag: str,
             history_window: int) -> dict:
    model = make_model(kind)
    baseline = None
    if kind == "dnn_sensor_baseline":
        earliest_batch = train.batch_id.min()
        baseline = train.loc[train.batch_id == earliest_batch, FEATURES].median()
    x_train = select_x(train, kind, baseline, history=train, history_window=history_window)
    if kind == "dnn_sensor_weighted":
        model.fit(x_train, train["gas_name"], mlp__sample_weight=difficult_sample_weights(train))
    else:
        model.fit(x_train, train["gas_name"])
    pred = model.predict(select_x(test, kind, baseline, history=train, history_window=history_window))
    row = {
        "experiment": tag, "model": kind,
        "train_batches": ",".join(map(str, sorted(train.batch_id.unique()))),
        "test_batch": int(test.batch_id.iloc[0]),
        "n_train": len(train), "n_test": len(test),
        "accuracy": accuracy_score(test.gas_name, pred),
        "macro_f1": f1_score(test.gas_name, pred, labels=GAS_NAMES, average="macro", zero_division=0),
    }
    recalls = recall_score(test.gas_name, pred, labels=GAS_NAMES, average=None, zero_division=0)
    row.update({f"recall_{gas}": float(v) for gas, v in zip(GAS_NAMES, recalls)})
    low_mask = test.concentration_ppm <= LOW_CONCENTRATION_PPM
    row["low_ppm_n"] = int(low_mask.sum())
    row["low_ppm_accuracy"] = accuracy_score(test.loc[low_mask, "gas_name"], pred[low_mask]) if low_mask.any() else np.nan
    cm = pd.DataFrame(confusion_matrix(test.gas_name, pred, labels=GAS_NAMES), index=GAS_NAMES, columns=GAS_NAMES)
    cm.to_csv(output / f"confusion_{tag}_{kind}_batch{row['test_batch']}.csv", encoding="utf-8-sig")

    # Concentration baseline: report separately, without claiming class-conditional calibration.
    reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    reg.fit(train[FEATURES], train.concentration_ppm)
    concentration_pred = reg.predict(test[FEATURES])
    row["concentration_mae"] = mean_absolute_error(test.concentration_ppm, concentration_pred)
    row["concentration_r2"] = r2_score(test.concentration_ppm, concentration_pred)
    return row

def main(csv_path: Path, output: Path, selected_models: list[str], experiments: list[str], test_batches: list[int],
         history_window: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"batch_id", "gas_name", "concentration_ppm", *FEATURES}
    missing = required.difference(df.columns)
    if missing: raise ValueError(f"CSV缺少字段: {sorted(missing)[:5]}")
    results = []
    models = selected_models
    # Fixed experiment required by the proposal: early batches -> each future batch.
    if "fixed" in experiments:
        for test_batch in test_batches:
            train, test = df[df.batch_id.isin([1, 2, 3])], df[df.batch_id == test_batch]
            for model in models: results.append(evaluate(train, test, model, output, "fixed_1to3", history_window))
    # Rolling experiment: all available history -> next unseen batch.
    if "rolling" in experiments:
        for test_batch in test_batches:
            train, test = df[df.batch_id < test_batch], df[df.batch_id == test_batch]
            for model in models: results.append(evaluate(train, test, model, output, "rolling", history_window))
    metrics = pd.DataFrame(results)
    metrics.to_csv(output / "cross_batch_metrics.csv", index=False, encoding="utf-8-sig")
    print(metrics[["experiment", "model", "test_batch", "accuracy", "macro_f1", "concentration_mae", "concentration_r2"]].to_string(index=False))
    print(f"\n结果目录: {output.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path(__file__).parent / "all_batches.csv")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "drift_results")
    ap.add_argument("--models", nargs="+", choices=["rf_sensor", "dnn_sensor", "dnn_sensor_time", "dnn_sensor_baseline", "dnn_sensor_history_baseline", "dnn_sensor_weighted"],
                    default=["rf_sensor", "dnn_sensor", "dnn_sensor_time", "dnn_sensor_baseline", "dnn_sensor_history_baseline", "dnn_sensor_weighted"])
    ap.add_argument("--experiments", nargs="+", choices=["fixed", "rolling"], default=["fixed", "rolling"])
    ap.add_argument("--test-batches", nargs="+", type=int, choices=range(4, 11), default=list(range(4, 11)))
    ap.add_argument("--history-window", type=int, default=3,
                    help="number of previous batches used for the sliding baseline (default: 3)")
    args = ap.parse_args(); main(args.csv, args.output, args.models, args.experiments, args.test_batches, args.history_window)
