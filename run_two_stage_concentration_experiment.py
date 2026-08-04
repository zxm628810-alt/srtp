"""Rolling two-stage DNN: classify gas first, then use a gas-specific regressor."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_drift_experiments import FEATURES, GAS_NAMES, select_x


def classifier() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("mlp", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), early_stopping=True, validation_fraction=.15,
        max_iter=400, random_state=42))])


def regressor() -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=400, random_state=42))]),
        transformer=StandardScaler(),
    )


def x(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    return select_x(frame, "dnn_sensor_history_baseline", history=history, history_window=3)


def evaluate(train: pd.DataFrame, test: pd.DataFrame, output: Path) -> dict:
    x_train, x_test = x(train, train), x(test, train)
    clf = classifier(); clf.fit(x_train, train.gas_name)
    predicted_gas = clf.predict(x_test)
    predictions = np.zeros(len(test))
    for gas in GAS_NAMES:
        mask_train = train.gas_name == gas
        mask_test = predicted_gas == gas
        if not mask_test.any():
            continue
        model = regressor(); model.fit(x_train.loc[mask_train], train.loc[mask_train, "concentration_ppm"])
        predictions[mask_test] = np.maximum(model.predict(x_test.loc[mask_test]), 0)
    details = test[["batch_id", "gas_name", "concentration_ppm"]].copy()
    details["predicted_gas"] = predicted_gas; details["predicted_ppm"] = predictions
    details["absolute_error_ppm"] = np.abs(details.concentration_ppm - predictions)
    batch = int(test.batch_id.iloc[0]); details.to_csv(output / f"predictions_two_stage_history_batch{batch}.csv", index=False, encoding="utf-8-sig")
    low = test.concentration_ppm <= 50
    result = {"model": "dnn_two_stage_history_w3", "test_batch": batch,
              "classification_accuracy": accuracy_score(test.gas_name, predicted_gas),
              "mae": mean_absolute_error(test.concentration_ppm, predictions),
              "r2": r2_score(test.concentration_ppm, predictions),
              "low_ppm_n": int(low.sum()),
              "low_ppm_mae": mean_absolute_error(test.loc[low, "concentration_ppm"], predictions[low]) if low.any() else np.nan}
    for gas in GAS_NAMES:
        mask = test.gas_name == gas
        result[f"mae_{gas}"] = mean_absolute_error(test.loc[mask, "concentration_ppm"], predictions[mask]) if mask.any() else np.nan
    return result


def main(csv: Path, output: Path, batches: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True); df = pd.read_csv(csv, encoding="utf-8-sig")
    rows = [evaluate(df[df.batch_id < batch], df[df.batch_id == batch], output) for batch in batches]
    table = pd.DataFrame(rows)
    metrics_path = output / "cross_batch_two_stage_metrics.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path, encoding="utf-8-sig")
        table = pd.concat([old[~old.test_batch.isin(table.test_batch)], table], ignore_index=True)
    table = table.sort_values("test_batch")
    table.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(table[["test_batch", "classification_accuracy", "mae", "r2", "low_ppm_mae"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "two_stage_concentration_results")
    parser.add_argument("--test-batches", nargs="+", type=int, choices=range(4, 11), default=list(range(4, 11)))
    args = parser.parse_args(); main(args.csv, args.output, args.test_batches)
