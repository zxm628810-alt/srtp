"""Rebuild concentration-regression metrics from saved rolling predictions."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from run_drift_experiments import GAS_NAMES


def main(output: Path) -> None:
    rows = []
    pattern = re.compile(r"predictions_(dnn_reg_.+)_batch(\d+)\.csv")
    for path in sorted(output.glob("predictions_dnn_reg_*_batch*.csv")):
        matched = pattern.fullmatch(path.name)
        if not matched:
            continue
        model, batch = matched.group(1), int(matched.group(2))
        frame = pd.read_csv(path, encoding="utf-8-sig")
        actual, predicted = frame.concentration_ppm, frame.predicted_ppm
        row = {
            "experiment": "rolling",
            "model": model,
            "train_batches": ",".join(map(str, range(1, batch))),
            "test_batch": batch,
            "n_train": "",
            "n_test": len(frame),
            "mae": mean_absolute_error(actual, predicted),
            "r2": r2_score(actual, predicted),
        }
        low = actual <= 50
        row["low_ppm_n"] = int(low.sum())
        row["low_ppm_mae"] = mean_absolute_error(actual[low], predicted[low]) if low.any() else float("nan")
        for gas in GAS_NAMES:
            mask = frame.gas_name == gas
            row[f"mae_{gas}"] = mean_absolute_error(actual[mask], predicted[mask]) if mask.any() else float("nan")
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values(["test_batch", "model"])
    metrics.to_csv(output / "cross_batch_concentration_metrics.csv", index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "concentration_drift_results")
    main(parser.parse_args().output)
