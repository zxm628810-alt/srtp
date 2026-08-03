"""Create report-ready figures for rolling DNN concentration regression."""
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "dnn_reg_sensor": "普通 DNN（128 维）",
    "dnn_reg_time": "DNN + 批次号",
    "dnn_reg_history_baseline": "DNN + 历史窗口 3",
}
COLORS = {"dnn_reg_sensor": "#4C78A8", "dnn_reg_time": "#F58518", "dnn_reg_history_baseline": "#54A24B"}


def configure() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def save_metric_curves(metrics: pd.DataFrame, figures: Path) -> None:
    for metric, ylabel, filename in [
        ("mae", "MAE（ppm，越低越好）", "浓度回归_MAE_滚动曲线.png"),
        ("r2", "R²（越高越好）", "浓度回归_R2_滚动曲线.png"),
        ("low_ppm_mae", "≤50 ppm 样本 MAE（ppm，越低越好）", "浓度回归_低浓度MAE_滚动曲线.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for model, group in metrics.groupby("model"):
            group = group.sort_values("test_batch")
            ax.plot(group.test_batch, group[metric], marker="o", linewidth=2.2,
                    label=LABELS[model], color=COLORS[model])
        if metric == "r2":
            ax.axhline(0, color="#777777", linewidth=1, linestyle="--")
        ax.set_xticks(range(4, 11)); ax.set_xlabel("测试 Batch")
        ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(figures / filename, dpi=220); plt.close(fig)


def save_gas_bars(metrics: pd.DataFrame, figures: Path) -> None:
    gases = ["Ethylene", "Ammonia", "Acetone", "Toluene"]
    cn = {"Ethylene": "乙烯", "Ammonia": "氨气", "Acetone": "丙酮", "Toluene": "甲苯"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, batch in zip(axes, [9, 10]):
        frame = metrics[metrics.test_batch == batch].set_index("model")
        x = range(len(gases)); width = .24
        for i, model in enumerate(LABELS):
            values = [frame.loc[model, f"mae_{gas}"] for gas in gases]
            ax.bar([p + (i - 1) * width for p in x], values, width, label=LABELS[model], color=COLORS[model])
        ax.set_title(f"Batch {batch}"); ax.set_xticks(list(x), [cn[g] for g in gases])
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("逐气体浓度 MAE（ppm，越低越好）")
    axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(figures / "浓度回归_逐气体MAE_Batch9_10.png", dpi=220); plt.close(fig)


def save_scatter(results: Path, figures: Path) -> None:
    models = ["dnn_reg_sensor", "dnn_reg_history_baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, model in zip(axes, models):
        frame = pd.read_csv(results / f"predictions_{model}_batch10.csv", encoding="utf-8-sig")
        ax.scatter(frame.concentration_ppm, frame.predicted_ppm, s=10, alpha=.35, color=COLORS[model], edgecolors="none")
        upper = max(frame.concentration_ppm.max(), frame.predicted_ppm.max()) * 1.03
        ax.plot([0, upper], [0, upper], "--", color="#333333", linewidth=1.2, label="理想预测 y=x")
        ax.set_title(LABELS[model]); ax.set_xlabel("真实浓度（ppm）")
        ax.grid(alpha=.25); ax.legend()
    axes[0].set_ylabel("预测浓度（ppm）")
    fig.tight_layout(); fig.savefig(figures / "浓度回归_真实预测散点_Batch10.png", dpi=220); plt.close(fig)


def main(results: Path, figures: Path) -> None:
    configure(); figures.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(results / "cross_batch_concentration_metrics.csv", encoding="utf-8-sig")
    save_metric_curves(metrics, figures); save_gas_bars(metrics, figures); save_scatter(results, figures)
    print(f"Figures saved to: {figures.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "concentration_drift_results")
    parser.add_argument("--figures", type=Path, default=Path(__file__).parent / "concentration_drift_figures")
    args = parser.parse_args(); main(args.results, args.figures)
