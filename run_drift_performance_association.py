"""Quantify feature-distribution drift and relate it to rolling model performance.

All drift statistics compare each Batch k only with batches earlier than k.
No gas class or concentration label is used to calculate drift strength.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from run_drift_experiments import FEATURES


ROOT = Path(__file__).parent


def drift_statistics(history: pd.DataFrame, current: pd.DataFrame, batch: int) -> tuple[dict, list[dict]]:
    scaler = StandardScaler().fit(history[FEATURES])
    history_z, current_z = scaler.transform(history[FEATURES]), scaler.transform(current[FEATURES])
    center_distance = float(np.linalg.norm(current_z.mean(axis=0)))
    pca = PCA(n_components=2, random_state=42).fit(history_z)
    pca_center_distance = float(np.linalg.norm(pca.transform(current_z).mean(axis=0) - pca.transform(history_z).mean(axis=0)))
    details = []
    for i, feature in enumerate(FEATURES):
        ks = ks_2samp(history_z[:, i], current_z[:, i]).statistic
        mean_shift = abs(current_z[:, i].mean())
        std_ratio = current_z[:, i].std(ddof=0) / (history_z[:, i].std(ddof=0) + 1e-12)
        details.append({"test_batch": batch, "feature": feature, "ks_statistic": ks,
                        "absolute_standardized_mean_shift": mean_shift, "standard_deviation_ratio": std_ratio})
    feature_table = pd.DataFrame(details)
    # Offline analysis only: compare the same gas against its own historical
    # samples, removing batch-to-batch class-composition as a confounder.
    conditional_rows = []
    for gas in sorted(current.gas_name.unique()):
        hist_g = history[history.gas_name == gas]
        cur_g = current[current.gas_name == gas]
        if hist_g.empty or cur_g.empty:
            continue
        gas_scaler = StandardScaler().fit(hist_g[FEATURES])
        hist_gz, cur_gz = gas_scaler.transform(hist_g[FEATURES]), gas_scaler.transform(cur_g[FEATURES])
        conditional_rows.append({"gas_name": gas,
                                 "center_distance": float(np.linalg.norm(cur_gz.mean(axis=0))),
                                 "mean_ks": float(np.mean([ks_2samp(hist_gz[:, i], cur_gz[:, i]).statistic for i in range(len(FEATURES))]))})
    conditional = pd.DataFrame(conditional_rows)
    summary = {"test_batch": batch, "n_history": len(history), "n_current": len(current),
               "standardized_center_distance": center_distance, "pca_center_distance": pca_center_distance,
               "mean_ks_statistic": feature_table.ks_statistic.mean(), "max_ks_statistic": feature_table.ks_statistic.max(),
               "mean_abs_standardized_shift": feature_table.absolute_standardized_mean_shift.mean(),
               "features_ks_ge_020": int((feature_table.ks_statistic >= .20).sum()),
               "class_conditional_center_distance": conditional.center_distance.mean(),
               "class_conditional_mean_ks": conditional.mean_ks.mean(),
               "class_conditional_gas_n": len(conditional)}
    return summary, details


def load_performance() -> pd.DataFrame:
    classification = pd.concat([
        pd.read_csv(ROOT / "window_selection_dev_w3" / "cross_batch_metrics.csv", encoding="utf-8-sig"),
        pd.read_csv(ROOT / "window_selection_final_w3" / "cross_batch_metrics.csv", encoding="utf-8-sig"),
    ], ignore_index=True)[["test_batch", "accuracy", "macro_f1"]]
    regression = pd.read_csv(ROOT / "concentration_drift_results" / "cross_batch_concentration_metrics.csv", encoding="utf-8-sig")
    regression = regression[regression.model == "dnn_reg_history_baseline"][["test_batch", "mae", "r2"]]
    boundary = pd.read_csv(ROOT / "decision_boundary_results" / "boundary_coverage_metrics.csv", encoding="utf-8-sig")
    boundary = boundary[boundary["set"] == "union"][["test_batch", "error_rate"]].rename(columns={"error_rate": "boundary_error_rate"})
    return classification.merge(regression, on="test_batch").merge(boundary, on="test_batch")


def correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    drift_columns = ["standardized_center_distance", "pca_center_distance", "mean_ks_statistic", "mean_abs_standardized_shift", "class_conditional_center_distance", "class_conditional_mean_ks"]
    performance_columns = ["accuracy", "macro_f1", "mae", "r2", "boundary_error_rate"]
    rows = []
    for drift in drift_columns:
        for metric in performance_columns:
            pearson = pearsonr(data[drift], data[metric]); spearman = spearmanr(data[drift], data[metric])
            rows.append({"drift_metric": drift, "performance_metric": metric,
                         "pearson_r": pearson.statistic, "pearson_p": pearson.pvalue,
                         "spearman_rho": spearman.statistic, "spearman_p": spearman.pvalue})
    return pd.DataFrame(rows)


def plot_associations(data: pd.DataFrame, figures: Path) -> None:
    metrics = [("accuracy", "分类准确率（越高越好）"), ("macro_f1", "分类 Macro-F1（越高越好）"),
               ("mae", "浓度 MAE ppm（越低越好）"), ("r2", "浓度 R²（越高越好）"),
               ("boundary_error_rate", "边界集错误率（越低越好）")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8)); axes = axes.ravel()
    for ax, (metric, label) in zip(axes, metrics):
        ax.scatter(data.class_conditional_mean_ks, data[metric], s=60, color="#4C78A8")
        for row in data.itertuples(): ax.annotate(f"B{row.test_batch}", (row.class_conditional_mean_ks, getattr(row, metric)), xytext=(4, 4), textcoords="offset points", fontsize=8)
        r = pearsonr(data.class_conditional_mean_ks, data[metric]).statistic
        ax.set_xlabel("同气体平均 KS 分布差异（越大表示漂移越强）"); ax.set_ylabel(label); ax.set_title(f"Pearson r = {r:.2f}"); ax.grid(alpha=.25)
    axes[-1].axis("off")
    fig.tight_layout(); fig.savefig(figures / "漂移强度与模型性能关联.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data.test_batch, data.mean_ks_statistic, marker="o", label="平均 KS 分布差异")
    ax.plot(data.test_batch, data.standardized_center_distance / data.standardized_center_distance.max(), marker="s", label="标准化中心距离（归一化）")
    ax.plot(data.test_batch, data.pca_center_distance / data.pca_center_distance.max(), marker="^", label="PCA 中心偏移（归一化）")
    ax.set_xticks(data.test_batch); ax.set_xlabel("测试 Batch"); ax.set_ylabel("漂移强度 / 归一化强度"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "各Batch漂移强度曲线.png", dpi=220); plt.close(fig)


def main(csv: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True); figures = output / "figures"; figures.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    data = pd.read_csv(csv, encoding="utf-8-sig"); summaries = []; details = []
    for batch in range(4, 11):
        summary, rows = drift_statistics(data[data.batch_id < batch], data[data.batch_id == batch], batch)
        summaries.append(summary); details.extend(rows)
    drift = pd.DataFrame(summaries); feature = pd.DataFrame(details)
    combined = drift.merge(load_performance(), on="test_batch")
    corr = correlation_table(combined)
    drift.to_csv(output / "batch_drift_summary.csv", index=False, encoding="utf-8-sig")
    feature.to_csv(output / "per_feature_distribution_drift.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(output / "drift_and_performance.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(output / "drift_performance_correlations.csv", index=False, encoding="utf-8-sig")
    plot_associations(combined, figures)
    print(combined.to_string(index=False)); print("\nCorrelation (mean KS):")
    print(corr[corr.drift_metric == "class_conditional_mean_ks"].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "drift_performance_association")
    args = parser.parse_args(); main(args.csv, args.output)
