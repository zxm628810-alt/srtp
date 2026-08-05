"""Plot selective vs all calibration comparison."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / "selective_calibration_results" / "selective_calibration_results.csv"
OUTPUT = ROOT / "selective_calibration_results" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(RESULTS, encoding="utf-8-sig")
batches = sorted(df["test_batch"].unique())
all_data = df[df["model"] == "dnn_all_calibrated"]
sel_data = df[df["model"] == "dnn_selective"]

# ---- Fig 1: Accuracy comparison bar chart ----
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
x = np.arange(len(batches))
w = 0.35
bars1 = ax.bar(x - w/2, all_data["accuracy"].values * 100, w, label="全部校准（128传感器）", color="#4C78A8", edgecolor="white")
bars2 = ax.bar(x + w/2, sel_data["accuracy"].values * 100, w, label="选择性校准（仅漂移传感器）", color="#F28E2B", edgecolor="white")

# Annotate difference
for i, batch in enumerate(batches):
    diff = (sel_data.iloc[i]["accuracy"] - all_data.iloc[i]["accuracy"]) * 100
    cal_pct = sel_data.iloc[i]["calibrated_pct"]
    color = "#2e7d32" if diff >= 0 else "#c62828"
    y = max(all_data.iloc[i]["accuracy"], sel_data.iloc[i]["accuracy"]) * 100 + 1.5
    ax.annotate(f"{diff:+.1f}%\n({cal_pct:.0f}%校准)", (x[i], y), ha="center", fontsize=8, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"Batch {b}" for b in batches])
ax.set_ylabel("分类准确率 (%)")
ax.set_title("逐通道选择性校准 vs 全部校准对比", fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "选择性校准准确率对比.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 2: Per-gas Recall comparison ----
fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=180)
axes = axes.ravel()
for i, gas in enumerate(GAS_NAMES):
    ax = axes[i]
    col = f"recall_{gas}"
    ax.plot(batches, all_data[col].values, "o-", color="#4C78A8", linewidth=2, markersize=7, label="全部校准")
    ax.plot(batches, sel_data[col].values, "s--", color="#F28E2B", linewidth=2, markersize=7, label="选择性")
    ax.set_title(gas, fontsize=11)
    ax.set_xticks(batches)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Recall")
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7)
fig.suptitle("逐气体 Recall：全部校准 vs 选择性校准", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT / "选择性校准逐气体Recall.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 3: Calibrated sensor count per batch ----
fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
cal_counts = sel_data["n_calibrated_features"].values
bar_colors = ["#4C78A8" if c >= 100 else "#F28E2B" for c in cal_counts]
bars = ax.bar(x, cal_counts, color=bar_colors, edgecolor="white")
for i, (c, pct) in enumerate(zip(cal_counts, sel_data["calibrated_pct"].values)):
    ax.annotate(f"{c}/128\n({pct:.0f}%)", (x[i], c + 2), ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"Batch {b}" for b in batches])
ax.set_ylabel("被校准的传感器数量")
ax.set_title("每个测试 Batch 中训练集内部 KS≥0.2 的传感器数", fontsize=12)
ax.set_ylim(0, 140)
ax.axhline(y=128, color="gray", linestyle="--", alpha=0.3)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "选择性校准传感器数量.png", bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"Figures saved: {OUTPUT.resolve()}")
