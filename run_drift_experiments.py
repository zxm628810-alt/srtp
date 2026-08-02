"""Strict temporal drift experiments for the UCI gas-sensor dataset.

Models:
  rf_sensor        RandomForest baseline using only 128 sensor features
  dnn_sensor       MLP neural network using only sensor features
  dnn_sensor_time  MLP using sensor features plus known batch/time feature

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

def make_model(kind: str):
    if kind == "rf_sensor":
        return RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu",
                               early_stopping=True, validation_fraction=.15,
                               max_iter=300, random_state=42))
    ])

def select_x(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    x = frame[FEATURES].copy()
    if kind == "dnn_sensor_time":
        # Device age/batch is assumed known at inference. It is not fitted using test labels.
        x["time_batch"] = frame["batch_id"].astype(float)
    return x

def evaluate(train: pd.DataFrame, test: pd.DataFrame, kind: str, output: Path, tag: str) -> dict:
    model = make_model(kind)
    model.fit(select_x(train, kind), train["gas_name"])
    pred = model.predict(select_x(test, kind))
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
    cm = pd.DataFrame(confusion_matrix(test.gas_name, pred, labels=GAS_NAMES), index=GAS_NAMES, columns=GAS_NAMES)
    cm.to_csv(output / f"confusion_{tag}_{kind}_batch{row['test_batch']}.csv", encoding="utf-8-sig")

    # Concentration baseline: report separately, without claiming class-conditional calibration.
    reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    reg.fit(train[FEATURES], train.concentration_ppm)
    concentration_pred = reg.predict(test[FEATURES])
    row["concentration_mae"] = mean_absolute_error(test.concentration_ppm, concentration_pred)
    row["concentration_r2"] = r2_score(test.concentration_ppm, concentration_pred)
    return row

def main(csv_path: Path, output: Path, selected_models: list[str], experiments: list[str], test_batches: list[int]) -> None:
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
            for model in models: results.append(evaluate(train, test, model, output, "fixed_1to3"))
    # Rolling experiment: all available history -> next unseen batch.
    if "rolling" in experiments:
        for test_batch in test_batches:
            train, test = df[df.batch_id < test_batch], df[df.batch_id == test_batch]
            for model in models: results.append(evaluate(train, test, model, output, "rolling"))
    metrics = pd.DataFrame(results)
    metrics.to_csv(output / "cross_batch_metrics.csv", index=False, encoding="utf-8-sig")
    print(metrics[["experiment", "model", "test_batch", "accuracy", "macro_f1", "concentration_mae", "concentration_r2"]].to_string(index=False))
    print(f"\n结果目录: {output.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path(__file__).parent / "all_batches.csv")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "drift_results")
    ap.add_argument("--models", nargs="+", choices=["rf_sensor", "dnn_sensor", "dnn_sensor_time"],
                    default=["rf_sensor", "dnn_sensor", "dnn_sensor_time"])
    ap.add_argument("--experiments", nargs="+", choices=["fixed", "rolling"], default=["fixed", "rolling"])
    ap.add_argument("--test-batches", nargs="+", type=int, choices=range(4, 11), default=list(range(4, 11)))
    args = ap.parse_args(); main(args.csv, args.output, args.models, args.experiments, args.test_batches)
