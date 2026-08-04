"""Additional strict rolling concentration experiments.

``dnn_reg_lowppm_weighted`` upweights training samples at or below 50 ppm.
``dnn_reg_history_residual`` uses out-of-fold residuals from earlier batches,
grouped by a separately predicted gas class.  The current test batch is never
used to estimate the correction.
"""
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

from run_drift_experiments import GAS_NAMES, select_x


LOW_PPM = 50.0
LOW_WEIGHT = 3.0


def features(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    return select_x(frame, "dnn_sensor_history_baseline", history=history, history_window=3)


def make_regressor() -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("mlp", MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), activation="relu", early_stopping=True,
            validation_fraction=.15, max_iter=400, random_state=42))]),
        transformer=StandardScaler(),
    )


def make_classifier() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("mlp", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu", early_stopping=True,
        validation_fraction=.15, max_iter=400, random_state=42))])


def metrics(test: pd.DataFrame, prediction: np.ndarray, model: str, batch: int) -> dict:
    low = test.concentration_ppm <= LOW_PPM
    row = {"model": model, "test_batch": batch, "mae": mean_absolute_error(test.concentration_ppm, prediction),
           "r2": r2_score(test.concentration_ppm, prediction), "low_ppm_n": int(low.sum()),
           "low_ppm_mae": mean_absolute_error(test.loc[low, "concentration_ppm"], prediction[low]) if low.any() else np.nan}
    for gas in GAS_NAMES:
        mask = test.gas_name == gas
        row[f"mae_{gas}"] = mean_absolute_error(test.loc[mask, "concentration_ppm"], prediction[mask]) if mask.any() else np.nan
    return row


def weighted_prediction(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_train, x_test = features(train, train), features(test, train)
    weights = np.where(train.concentration_ppm.to_numpy() <= LOW_PPM, LOW_WEIGHT, 1.0)
    model = make_regressor()
    model.fit(x_train, train.concentration_ppm, mlp__sample_weight=weights)
    return np.maximum(model.predict(x_test), 0)


def residual_biases(train: pd.DataFrame) -> pd.Series:
    """Estimate class-conditional residuals using only earlier-batch OOF outputs."""
    records = []
    for held_batch in sorted(train.batch_id.unique()):
        fit = train[train.batch_id < held_batch]
        held = train[train.batch_id == held_batch]
        if fit.empty:
            continue
        x_fit, x_held = features(fit, fit), features(held, fit)
        reg, clf = make_regressor(), make_classifier()
        reg.fit(x_fit, fit.concentration_ppm); clf.fit(x_fit, fit.gas_name)
        out = pd.DataFrame({"predicted_gas": clf.predict(x_held),
                            "residual": held.concentration_ppm.to_numpy() - np.maximum(reg.predict(x_held), 0)})
        records.append(out)
    if not records:
        return pd.Series(0.0, index=GAS_NAMES)
    return pd.concat(records).groupby("predicted_gas").residual.mean().reindex(GAS_NAMES, fill_value=0.0)


def residual_prediction(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    x_train, x_test = features(train, train), features(test, train)
    reg, clf = make_regressor(), make_classifier()
    reg.fit(x_train, train.concentration_ppm); clf.fit(x_train, train.gas_name)
    predicted_gas = clf.predict(x_test)
    bias = residual_biases(train)
    correction = pd.Series(predicted_gas).map(bias).fillna(0).to_numpy()
    return np.maximum(reg.predict(x_test) + correction, 0), predicted_gas, bias


def main(csv: Path, output: Path, batches: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True); data = pd.read_csv(csv, encoding="utf-8-sig")
    rows = []
    for batch in batches:
        train, test = data[data.batch_id < batch], data[data.batch_id == batch]
        weighted = weighted_prediction(train, test)
        rows.append(metrics(test, weighted, "dnn_reg_lowppm_weighted", batch))
        residual, gas, bias = residual_prediction(train, test)
        row = metrics(test, residual, "dnn_reg_history_residual", batch)
        row["residual_classifier_accuracy"] = accuracy_score(test.gas_name, gas)
        rows.append(row)
        pd.DataFrame({"gas": bias.index, "historical_residual_correction_ppm": bias.values}).to_csv(
            output / f"historical_residual_bias_batch{batch}.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(rows)
    metrics_path = output / "cross_batch_lowppm_residual_metrics.csv"
    if metrics_path.exists():
        previous = pd.read_csv(metrics_path, encoding="utf-8-sig")
        table = pd.concat([previous[~previous.test_batch.isin(table.test_batch)], table], ignore_index=True)
    table = table.sort_values(["test_batch", "model"])
    table.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(table[["model", "test_batch", "mae", "r2", "low_ppm_mae"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "lowppm_residual_results")
    parser.add_argument("--test-batches", nargs="+", type=int, choices=range(4, 11), default=list(range(4, 11)))
    args = parser.parse_args(); main(args.csv, args.output, args.test_batches)
