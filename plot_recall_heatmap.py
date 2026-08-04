"""Per-gas recall heatmap across batches 4-10 for all 7 rolling models.

Reads the master comparison CSV, extracts per-gas recall columns,
and renders a heatmap grid: rows=gas, columns=batch, color=recall.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
OUTPUT = ROOT / "figures"

GAS_LABELS = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]

# ---- data sources ----
SOURCES = {
    "RF":                  ROOT / "drift_results_rf_rolling" / "cross_batch_metrics.csv",
    "DNN sensor":          ROOT / "drift_results_dnn_sensor_rolling.csv",
    "DNN + batch ID":      ROOT / "drift_results_dnn_time_rolling" / "cross_batch_metrics.csv",
    "DNN + fixed baseline": ROOT / "drift_results_dnn_history_baseline_rolling_4" / "cross_batch_metrics.csv",
    "DNN + sliding baseline": ROOT / "drift_results_dnn_history_baseline_rolling_4" / "cross_batch_metrics.csv",
    "DNN + weights":       ROOT / "drift_results_dnn_weighted_rolling_4" / "cross_batch_metrics.csv",
    "Fusion DNN":          ROOT / "drift_results_fusion_rolling" / "cross_batch_metrics.csv",
}


def load_recalls(csv_path: Path, model_name: str) -> pd.DataFrame:
    """Read a cross_batch_metrics.csv and return a DataFrame with test_batch + per-gas recalls."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # drift_results_dnn_sensor_rolling.csv has extra quoting – strip
    df.columns = [c.strip().strip('"') for c in df.columns]

    rows = []
    for _, row in df.iterrows():
        batch = int(row["test_batch"])
        for gas in GAS_LABELS:
            col = f"recall_{gas}"
            if col in row:
                rows.append({"model": model_name, "test_batch": batch, "gas": gas, "recall": float(row[col])})
    return pd.DataFrame(rows)


def main():
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    all_data = []
    for model_name, path in SOURCES.items():
        if path.exists():
            all_data.append(load_recalls(path, model_name))
        else:
            print(f"[SKIP] {path} not found for {model_name}")
    df = pd.concat(all_data, ignore_index=True)

    models = df["model"].unique()
    batches = sorted(df["test_batch"].unique())

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(3.2 * n_models, 6.5), dpi=180,
                             sharey=True, gridspec_kw={"wspace": 0.08})

    if n_models == 1:
        axes = [axes]

    vmin, vmax = 0.0, 1.0

    for ax, model_name in zip(axes, models):
        sub = df[df["model"] == model_name].pivot_table(
            index="gas", columns="test_batch", values="recall", aggfunc="first"
        )
        # ensure consistent row order
        sub = sub.reindex(index=GAS_LABELS, columns=batches)

        sns.heatmap(sub, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=vmin, vmax=vmax, linewidths=0.8, linecolor="white",
                    cbar=(ax == axes[-1]),  # only show colorbar on last subplot
                    cbar_kws={"shrink": 0.75, "label": "Recall"} if ax == axes[-1] else {},
                    annot_kws={"fontsize": 8})
        ax.set_title(model_name, fontsize=11, pad=8)
        ax.set_xlabel("Test Batch")
        if ax == axes[0]:
            ax.set_ylabel("Gas")
        else:
            ax.set_ylabel("")

    fig.suptitle("Per-Gas Recall Across Future Batches (Rolling Validation)", fontsize=14, y=1.01)
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / "recall_heatmap_all_models.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Figure saved: {OUTPUT / 'recall_heatmap_all_models.png'}")

    # ---- also make a single large heatmap for the fusion DNN only ----
    fusion_df = df[df["model"] == "Fusion DNN"].pivot_table(
        index="gas", columns="test_batch", values="recall", aggfunc="first"
    )
    fusion_df = fusion_df.reindex(index=GAS_LABELS, columns=batches)

    fig2, ax2 = plt.subplots(figsize=(8, 5.5), dpi=180)
    sns.heatmap(fusion_df, ax=ax2, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=vmin, vmax=vmax, linewidths=1.2, linecolor="white",
                cbar_kws={"shrink": 0.8, "label": "Recall"}, annot_kws={"fontsize": 11})
    ax2.set_title("Fusion DNN (PyTorch): Per-Gas Recall by Future Batch", fontsize=13)
    ax2.set_xlabel("Test Batch"); ax2.set_ylabel("Gas")
    fig2.tight_layout()
    fig2.savefig(OUTPUT / "recall_heatmap_fusion_dnn.png", bbox_inches="tight", dpi=200)
    plt.close(fig2)
    print(f"Figure saved: {OUTPUT / 'recall_heatmap_fusion_dnn.png'}")

    # ---- print worst recalls ----
    print("\n=== LOWEST RECALLS PER MODEL ===")
    for model_name in models:
        sub = df[df["model"] == model_name]
        worst = sub.loc[sub["recall"].idxmin()]
        print(f"  {model_name}: Batch {int(worst['test_batch'])}, {worst['gas']} = {worst['recall']:.3f}")


if __name__ == "__main__":
    main()
