"""Decision-boundary coverage test under strict rolling temporal evaluation.

For every future Batch k, the DNN is trained with Batch 1..k-1 only.  A test
sample is labelled "near boundary" before its true label is inspected when at
least one of these frozen rules holds:
1) DNN uncertainty: max class probability < 0.70 OR top-1/top-2 margin < 0.25.
2) PCA geometry: after PCA fitted on historical data only, the relative gap
   between distances to its two nearest historical class centroids is < 0.15.
The union supplies the coverage set; labels are used only afterwards to count
errors, recalls and confusion pairs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_drift_experiments import FEATURES, GAS_NAMES, make_model, select_x


MAX_PROBABILITY_THRESHOLD = .70
MARGIN_THRESHOLD = .25
PCA_GAP_THRESHOLD = .15


def pca_positions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    scaler = StandardScaler(); pca = PCA(n_components=2, random_state=42)
    train_xy = pca.fit_transform(scaler.fit_transform(train[FEATURES]))
    test_xy = pca.transform(scaler.transform(test[FEATURES]))
    centroids = {gas: train_xy[train.gas_name.to_numpy() == gas].mean(axis=0) for gas in GAS_NAMES}
    return train_xy, test_xy, centroids


def boundary_frame(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    model = make_model("dnn_sensor_history_baseline")
    x_train = select_x(train, "dnn_sensor_history_baseline", history=train, history_window=3)
    x_test = select_x(test, "dnn_sensor_history_baseline", history=train, history_window=3)
    model.fit(x_train, train.gas_name)
    probabilities = model.predict_proba(x_test)
    order = np.argsort(probabilities, axis=1)
    predicted = model.classes_[order[:, -1]]
    max_probability = probabilities[np.arange(len(test)), order[:, -1]]
    margin = max_probability - probabilities[np.arange(len(test)), order[:, -2]]
    _, xy, centroids = pca_positions(train, test)
    distances = np.column_stack([np.linalg.norm(xy - centroids[gas], axis=1) for gas in GAS_NAMES])
    distance_order = np.argsort(distances, axis=1)
    d1 = distances[np.arange(len(test)), distance_order[:, 0]]
    d2 = distances[np.arange(len(test)), distance_order[:, 1]]
    pca_gap = (d2 - d1) / (d2 + d1 + 1e-12)
    probability_boundary = (max_probability < MAX_PROBABILITY_THRESHOLD) | (margin < MARGIN_THRESHOLD)
    pca_boundary = pca_gap < PCA_GAP_THRESHOLD
    boundary = probability_boundary | pca_boundary
    out = test[["batch_id", "gas_name", "concentration_ppm"]].copy()
    out["predicted_gas"] = predicted; out["correct"] = out.gas_name.eq(out.predicted_gas)
    out["max_probability"] = max_probability; out["probability_margin"] = margin
    out["pca_nearest_class"] = np.array(GAS_NAMES)[distance_order[:, 0]]; out["pca_gap_ratio"] = pca_gap
    out["probability_boundary"] = probability_boundary; out["pca_boundary"] = pca_boundary; out["boundary_union"] = boundary
    out["pca_x"] = xy[:, 0]; out["pca_y"] = xy[:, 1]
    return out, xy, np.array([centroids[gas] for gas in GAS_NAMES])


def summarize(frame: pd.DataFrame, batch: int) -> list[dict]:
    rows = []
    for name, mask in [("probability", frame.probability_boundary), ("pca_geometry", frame.pca_boundary),
                       ("union", frame.boundary_union), ("non_boundary", ~frame.boundary_union)]:
        subset = frame[mask]
        rows.append({"test_batch": batch, "set": name, "n_samples": len(subset),
                     "share": len(subset) / len(frame), "accuracy": accuracy_score(subset.gas_name, subset.predicted_gas) if len(subset) else np.nan,
                     "error_rate": 1 - accuracy_score(subset.gas_name, subset.predicted_gas) if len(subset) else np.nan,
                     "low_ppm_n": int((subset.concentration_ppm <= 50).sum())})
    return rows


def plot_batch(frame: pd.DataFrame, centroids: np.ndarray, batch: int, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    normal = frame[~frame.boundary_union]; boundary = frame[frame.boundary_union]
    ax.scatter(normal.pca_x, normal.pca_y, s=13, alpha=.38, c="#4C78A8", label="非边界样本")
    ax.scatter(boundary.pca_x, boundary.pca_y, s=24, alpha=.82, marker="x", c="#E45756", label="边界覆盖样本")
    ax.scatter(centroids[:, 0], centroids[:, 1], s=85, marker="*", c="#222222", label="历史类别中心")
    for point, gas in zip(centroids, GAS_NAMES): ax.annotate(gas, point, xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title(f"Batch {batch}：PCA 决策边界覆盖样本")
    ax.set_xlabel("PCA 1（仅由历史训练 Batch 拟合）"); ax.set_ylabel("PCA 2")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures / f"决策边界覆盖_PCA_Batch{batch}.png", dpi=220); plt.close(fig)


def main(csv: Path, output: Path, batches: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True); figures = output / "figures"; figures.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    data = pd.read_csv(csv, encoding="utf-8-sig"); metrics = []; pairs = []
    for batch in batches:
        train, test = data[data.batch_id < batch], data[data.batch_id == batch]
        frame, _, centroids = boundary_frame(train, test)
        frame.to_csv(output / f"boundary_samples_batch{batch}.csv", index=False, encoding="utf-8-sig")
        metrics.extend(summarize(frame, batch)); plot_batch(frame, centroids, batch, figures)
        errors = frame[frame.boundary_union & ~frame.correct]
        pairs.extend({"test_batch": batch, "actual_gas": a, "predicted_gas": p, "errors": n}
                     for (a, p), n in errors.groupby(["gas_name", "predicted_gas"]).size().items())
    pd.DataFrame(metrics).to_csv(output / "boundary_coverage_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairs).sort_values(["test_batch", "errors"], ascending=[True, False]).to_csv(output / "boundary_error_pairs.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(metrics); print(table[table["set"].isin(["union", "non_boundary"])].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "decision_boundary_results")
    parser.add_argument("--test-batches", nargs="+", type=int, choices=range(4, 11), default=list(range(4, 11)))
    args = parser.parse_args(); main(args.csv, args.output, args.test_batches)
