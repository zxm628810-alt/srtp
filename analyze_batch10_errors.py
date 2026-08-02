"""Analyze class confusions and concentration-dependent errors on future Batch 10."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from run_drift_experiments import FEATURES, GAS_NAMES, make_model, select_x


ROOT = Path(__file__).parent
OUTPUT = ROOT / "figures"
DATA = ROOT / "all_batches.csv"
CONFUSION_FILES = {
    "RandomForest": ROOT / "drift_results_rf_rolling" / "confusion_rolling_rf_sensor_batch10.csv",
    "DNN (sensor only)": ROOT / "drift_results_dnn_sensor_rolling_10" / "confusion_rolling_dnn_sensor_batch10.csv",
    "DNN (+ batch ID)": ROOT / "drift_results_dnn_time_rolling_10" / "confusion_rolling_dnn_sensor_time_batch10.csv",
    "DNN (+ baseline features)": ROOT / "drift_results_dnn_baseline_rolling_10" / "confusion_rolling_dnn_sensor_baseline_batch10.csv",
}


def confusion_figure() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=160, layout="constrained")
    for ax, (name, path) in zip(axes.ravel(), CONFUSION_FILES.items()):
        cm = pd.read_csv(path, index_col=0, encoding="utf-8-sig").reindex(index=GAS_NAMES, columns=GAS_NAMES, fill_value=0)
        normalized = cm.div(cm.sum(axis=1).replace(0, 1), axis=0).to_numpy()
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set(title=name, xlabel="Predicted gas", ylabel="True gas")
        ax.set_xticks(range(len(GAS_NAMES)), GAS_NAMES, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(GAS_NAMES)), GAS_NAMES, fontsize=8)
        for i in range(len(GAS_NAMES)):
            for j in range(len(GAS_NAMES)):
                value = normalized[i, j]
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if value > .55 else "black")
    fig.savefig(OUTPUT / "error_confusion_matrices_batch10.png", bbox_inches="tight")
    plt.close(fig)


def prediction_and_concentration_analysis() -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    train, test = df[df.batch_id < 10], df[df.batch_id == 10].copy()
    kind = "dnn_sensor_time"
    model = make_model(kind)
    model.fit(select_x(train, kind), train.gas_name)
    test["predicted_gas"] = model.predict(select_x(test, kind))
    test["is_correct"] = test.gas_name.eq(test.predicted_gas)
    test.to_csv(OUTPUT / "error_predictions_batch10_dnn_time.csv", index=False, encoding="utf-8-sig")

    # Fixed, interpretable ppm intervals. Empty cells remain blank rather than being treated as zero.
    bins = [0, 10, 50, 100, 250, 1000.001]
    labels = ["≤10", "10-50", "50-100", "100-250", "250-1000"]
    test["ppm_interval"] = pd.cut(test.concentration_ppm, bins=bins, labels=labels, include_lowest=True)
    summary = (test.groupby(["gas_name", "ppm_interval"], observed=False)
                    .agg(samples=("is_correct", "size"), accuracy=("is_correct", "mean"))
                    .reset_index())
    summary.to_csv(OUTPUT / "error_by_gas_and_ppm_batch10.csv", index=False, encoding="utf-8-sig")

    accuracy_grid = summary.pivot(index="gas_name", columns="ppm_interval", values="accuracy").reindex(index=GAS_NAMES, columns=labels)
    count_grid = summary.pivot(index="gas_name", columns="ppm_interval", values="samples").reindex(index=GAS_NAMES, columns=labels)
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
    image = ax.imshow(accuracy_grid.to_numpy(dtype=float), vmin=0, vmax=1, cmap="YlOrRd")
    ax.set(title="DNN (+ batch ID): Batch 10 accuracy by gas and concentration", xlabel="True concentration (ppm)", ylabel="True gas")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(GAS_NAMES)), GAS_NAMES)
    for i in range(len(GAS_NAMES)):
        for j in range(len(labels)):
            value, count = accuracy_grid.iloc[i, j], count_grid.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.0%}\n(n={int(count)})", ha="center", va="center", fontsize=8,
                        color="white" if value < .45 else "black")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="gray")
    fig.colorbar(image, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(OUTPUT / "error_accuracy_by_gas_ppm_batch10.png", bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(test.gas_name, test.predicted_gas, labels=GAS_NAMES)
    pairs = []
    for i, actual in enumerate(GAS_NAMES):
        for j, predicted in enumerate(GAS_NAMES):
            if i != j and cm[i, j]:
                pairs.append({"true_gas": actual, "predicted_as": predicted, "errors": int(cm[i, j]),
                              "true_gas_error_rate": float(cm[i, j] / cm[i].sum())})
    pd.DataFrame(pairs).sort_values("errors", ascending=False).to_csv(
        OUTPUT / "top_misclassifications_batch10_dnn_time.csv", index=False, encoding="utf-8-sig"
    )
    print("Per-gas accuracy:")
    print(test.groupby("gas_name").is_correct.agg(["mean", "size"]).to_string())
    print("\nTop misclassifications:")
    print(pd.DataFrame(pairs).sort_values("errors", ascending=False).head(10).to_string(index=False))


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    confusion_figure()
    prediction_and_concentration_analysis()
    print(f"Error-analysis outputs saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
