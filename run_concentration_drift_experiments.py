"""Strict rolling temporal evaluation for DNN gas-concentration regression.

Models:
  dnn_reg_sensor            128 raw sensor features
  dnn_reg_time              raw features plus known batch/time feature
  dnn_reg_history_baseline  raw features plus 3-batch sliding baseline features

The current test batch is never used to fit a feature scaler, target scaler or
history baseline. Concentration predictions are clipped at zero because ppm
cannot be negative.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_drift_experiments import FEATURES, GAS_NAMES, select_x


INPUT_KIND = {
    "dnn_reg_sensor": "dnn_sensor",
    "dnn_reg_time": "dnn_sensor_time",
    "dnn_reg_history_baseline": "dnn_sensor_history_baseline",
}
LOW_CONCENTRATION_PPM = 50.0


def make_regressor() -> TransformedTargetRegressor:
    """Use train-only scaling for inputs and target concentration."""
    regressor = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            early_stopping=True, validation_fraction=.15, max_iter=400,
            random_state=42,
        )),
    ])
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def input_features(frame: pd.DataFrame, model_name: str, train_history: pd.DataFrame) -> pd.DataFrame:
    return select_x(
        frame,
        INPUT_KIND[model_name],
        history=train_history,
        history_window=3,
    )


def evaluate(train: pd.DataFrame, test: pd.DataFrame, model_name: str, output: Path) -> dict:
    model = make_regressor()
    x_train = input_features(train, model_name, train)
    x_test = input_features(test, model_name, train)
    model.fit(x_train, train.concentration_ppm)
    prediction = np.maximum(model.predict(x_test), 0.0)

    result = {
        "experiment": "rolling",
        "model": model_name,
        "train_batches": ",".join(map(str, sorted(train.batch_id.unique()))),
        "test_batch": int(test.batch_id.iloc[0]),
        "n_train": len(train),
        "n_test": len(test),
        "mae": mean_absolute_error(test.concentration_ppm, prediction),
        "r2": r2_score(test.concentration_ppm, prediction),
    }
    low_mask = test.concentration_ppm <= LOW_CONCENTRATION_PPM
    result["low_ppm_n"] = int(low_mask.sum())
    result["low_ppm_mae"] = mean_absolute_error(test.loc[low_mask, "concentration_ppm"], prediction[low_mask]) if low_mask.any() else np.nan
    for gas in GAS_NAMES:
        gas_mask = test.gas_name == gas
        result[f"mae_{gas}"] = mean_absolute_error(test.loc[gas_mask, "concentration_ppm"], prediction[gas_mask]) if gas_mask.any() else np.nan

    details = test[["batch_id", "gas_name", "concentration_ppm"]].copy()
    details["predicted_ppm"] = prediction
    details["absolute_error_ppm"] = np.abs(details.concentration_ppm - details.predicted_ppm)
    details.to_csv(output / f"predictions_{model_name}_batch{result['test_batch']}.csv", index=False, encoding="utf-8-sig")
    return result


def main(csv_path: Path, output: Path, models: list[str], batches: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"batch_id", "gas_name", "concentration_ppm", *FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)[:5]}")

    results = []
    for batch in batches:
        train, test = df[df.batch_id < batch], df[df.batch_id == batch]
        for model_name in models:
            results.append(evaluate(train, test, model_name, output))
    metrics = pd.DataFrame(results)
    metrics.to_csv(output / "cross_batch_concentration_metrics.csv", index=False, encoding="utf-8-sig")
    print(metrics[["model", "test_batch", "mae", "r2", "low_ppm_mae"]].to_string(index=False))
    print(f"\nResults saved to: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "concentration_drift_results")
    parser.add_argument("--models", nargs="+", choices=list(INPUT_KIND), default=list(INPUT_KIND))
    parser.add_argument("--test-batches", nargs="+", type=int, choices=range(4, 11), default=list(range(4, 11)))
    args = parser.parse_args()
    main(args.csv, args.output, args.models, args.test_batches)
