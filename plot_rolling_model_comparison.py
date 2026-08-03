"""Create report-ready curves for the four-model rolling drift experiment."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).parent
DATA = ROOT / "滚动验证模型对比.csv"
OUTPUT = ROOT / "figures"

MODEL_COLUMNS = {
    "RandomForest": ("RF_准确率", "RF_macroF1"),
    "DNN (sensor only)": ("DNN传感器_准确率", "DNN传感器_macroF1"),
    "DNN (+ batch ID)": ("DNN传感器加批次号_准确率", "DNN传感器加批次号_macroF1"),
    "DNN (+ fixed baseline)": ("DNN固定基线_准确率", "DNN固定基线_macroF1"),
    "DNN (+ sliding history baseline)": ("DNN滑动历史基线_准确率", "DNN滑动历史基线_macroF1"),
    "DNN (+ difficult-sample weights)": ("DNN困难样本加权_准确率", "DNN困难样本加权_macroF1"),
}
MARKERS = ["o", "s", "^", "D", "P", "X"]


def draw(metric_index: int, title: str, y_label: str, filename: str) -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
    for ((name, columns), marker) in zip(MODEL_COLUMNS.items(), MARKERS):
        values = df[columns[metric_index]]
        ax.plot(df["测试批次"], values, marker=marker, linewidth=2.2, markersize=6, label=name)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Future test batch (trained only on earlier batches)")
    ax.set_ylabel(y_label)
    ax.set_xticks(df["测试批次"])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    draw(0, "Rolling temporal validation: accuracy by future batch", "Accuracy", "rolling_accuracy.png")
    draw(1, "Rolling temporal validation: Macro-F1 by future batch", "Macro-F1", "rolling_macro_f1.png")
    print(f"Figures saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
