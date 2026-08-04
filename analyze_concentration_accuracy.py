"""Accuracy by concentration interval for gas classification under rolling validation.

Bins every test sample by its true concentration (ppm) into intervals:
  ≤25, 25-50, 50-100, 100-200, >200

For each model and each future batch, computes per-interval accuracy.
Produces a summary CSV and a faceted bar chart or line chart.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_CSV = ROOT / "all_batches.csv"
OUTPUT = ROOT / "figures"
OUTPUT_DIR = ROOT / "concentration_interval_analysis"

INTERVALS = [
    (0, 25, "<=25 ppm"),
    (25, 50, "25-50 ppm"),
    (50, 100, "50-100 ppm"),
    (100, 200, "100-200 ppm"),
    (200, 1e9, ">200 ppm"),
]

FEATURES = [f"feature_{i}" for i in range(1, 129)]
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def interval_label(ppm: float) -> str:
    for lo, hi, label in INTERVALS:
        if lo <= ppm <= hi:
            return label
    return "unknown"


def evaluate_model(train: pd.DataFrame, test: pd.DataFrame, model_name: str,
                    history: pd.DataFrame = None, results_dir: Path = None) -> pd.DataFrame:
    """Train an sklearn MLP on train, predict on test, return per-sample predictions."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from run_drift_experiments import select_x, FEATURES, GAS_NAMES

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu",
                               early_stopping=True, validation_fraction=.15,
                               max_iter=300, random_state=42))
    ])

    kwargs = {}
    if model_name in ("dnn_sensor_history_baseline", "dnn_sensor_history_robust"):
        kwargs["history"] = history
        kwargs["history_window"] = 3
    x_train = select_x(train, model_name, **kwargs)
    if model_name in ("dnn_sensor_history_baseline", "dnn_sensor_history_robust"):
        x_test = select_x(test, model_name, history=history, history_window=3)
    else:
        x_test = select_x(test, model_name)

    pipe.fit(x_train, train["gas_name"])
    pred = pipe.predict(x_test)

    out = test[["batch_id", "gas_name", "concentration_ppm"]].copy()
    out["test_batch"] = out["batch_id"]  # alias for consistent grouping
    out["predicted"] = pred
    out["correct"] = out["gas_name"] == out["predicted"]
    out["interval"] = out["concentration_ppm"].apply(interval_label)
    out["model"] = model_name
    out["n_train"] = len(train)
    return out


def main():
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")

    # ---- model list: use 4 representative models ----
    models_to_run = {
        "dnn_sensor": "DNN (sensor only)",
        "dnn_sensor_time": "DNN + batch ID",
        "dnn_sensor_history_baseline": "DNN + sliding baseline",
        "dnn_sensor_history_robust": "DNN + IQR enhanced",
    }

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_preds = []

    for model_key, model_label in models_to_run.items():
        print(f"\n--- {model_label} ---")
        for test_batch in range(4, 11):
            train = df[df["batch_id"] < test_batch]
            test = df[df["batch_id"] == test_batch]
            print(f"  Batch {test_batch}: train={len(train)}, test={len(test)}")

            pred_df = evaluate_model(train, test, model_key, history=train, results_dir=OUTPUT_DIR)
            pred_df["model_label"] = model_label
            all_preds.append(pred_df)

    # Merge all predictions
    preds = pd.concat(all_preds, ignore_index=True)

    # ---- Summary: accuracy by model × batch × interval ----
    summary = preds.groupby(["model_label", "test_batch", "interval"]).agg(
        n_samples=("correct", "count"),
        accuracy=("correct", "mean"),
    ).reset_index()
    summary["accuracy"] = summary["accuracy"].round(4)

    summary.to_csv(OUTPUT_DIR / "concentration_interval_accuracy.csv", index=False, encoding="utf-8-sig")
    print(f"\nSummary saved: {OUTPUT_DIR / 'concentration_interval_accuracy.csv'}")

    # ---- Print batch 10 table ----
    print("\n\n=== BATCH 10: Accuracy by concentration interval ===")
    b10 = summary[summary["test_batch"] == 10].pivot_table(
        index="interval", columns="model_label", values="accuracy", aggfunc="first"
    )
    # reorder intervals
    interval_order = [lab for _, _, lab in INTERVALS]
    b10 = b10.reindex(interval_order)
    print(b10.to_string())

    # ---- Plot: line chart, accuracy vs batch, faceted by interval ----
    OUTPUT.mkdir(exist_ok=True)

    for interval_name in interval_order:
        sub = summary[summary["interval"] == interval_name]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        for mdl in sub["model_label"].unique():
            s = sub[sub["model_label"] == mdl]
            ax.plot(s["test_batch"], s["accuracy"], marker="o", linewidth=2.2, markersize=6, label=mdl)

        ax.set_title(f"Classification accuracy for {interval_name} samples", fontsize=13)
        ax.set_xlabel("Test Batch"); ax.set_ylabel("Accuracy")
        ax.set_xticks(range(4, 11)); ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        safe_name = interval_name.replace('>', 'gt').replace('<=', 'le').replace(' ', '_')
        fig.savefig(OUTPUT / f"concentration_accuracy_{safe_name}.png",
                    bbox_inches="tight", dpi=180)
        plt.close(fig)

    # ---- Bar chart comparing intervals at batch 10 ----
    fig2, ax2 = plt.subplots(figsize=(10, 5.5), dpi=180)
    b10_long = preds[(preds["test_batch"] == 10)].copy()
    # compute per-model per-interval accuracy
    bar_data = b10_long.groupby(["model_label", "interval"])["correct"].mean().reset_index()
    bar_data.columns = ["Model", "Interval", "Accuracy"]

    import seaborn as sns
    sns.barplot(data=bar_data, x="Interval", y="Accuracy", hue="Model", ax=ax2,
                palette="Set2", edgecolor="gray", linewidth=0.8)
    ax2.set_title("Batch 10: Classification Accuracy by Concentration Interval", fontsize=14)
    ax2.set_ylabel("Accuracy"); ax2.set_xlabel("Concentration Interval")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter("{x:.0%}")
    ax2.legend(fontsize=8, loc="lower left")
    fig2.tight_layout()
    fig2.savefig(OUTPUT / "concentration_accuracy_batch10_bars.png", bbox_inches="tight", dpi=200)
    plt.close(fig2)
    print(f"\nFigures saved: {OUTPUT / 'concentration_accuracy_*.png'}")

    # ---- global low vs high concentration summary ----
    preds["low_ppm"] = preds["concentration_ppm"] <= 50
    global_summary = preds.groupby(["model_label", "test_batch", "low_ppm"])["correct"].mean().reset_index()
    global_summary["accuracy"] = global_summary["correct"].round(4)
    global_summary.to_csv(OUTPUT_DIR / "low_vs_high_concentration_accuracy.csv", index=False, encoding="utf-8-sig")

    print("\n=== OVERALL: ≤50 ppm vs >50 ppm accuracy (Batch 10) ===")
    b10_g = global_summary[global_summary["test_batch"] == 10]
    for mdl in b10_g["model_label"].unique():
        s = b10_g[b10_g["model_label"] == mdl]
        lo = s[s["low_ppm"] == True]["accuracy"].values
        hi = s[s["low_ppm"] == False]["accuracy"].values
        print(f"  {mdl}:  ≤50 ppm = {lo[0]:.3f}  |  >50 ppm = {hi[0]:.3f}")


if __name__ == "__main__":
    main()
