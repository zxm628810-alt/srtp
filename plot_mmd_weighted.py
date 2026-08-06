"""Plot weighted MMD experiment results."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / "mmd_weighted_results" / "mmd_weighted_results.csv"
DEV_SUMMARY = ROOT / "mmd_weighted_results" / "dev_selection_summary.csv"
OUTPUT = ROOT / "mmd_weighted_results" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(RESULTS, encoding="utf-8-sig")
dev = df[df["phase"] == "dev"]
final = df[df["phase"] == "final"]

# ---- Fig 1: Dev phase selection ----
dev_summary = pd.read_csv(DEV_SUMMARY, encoding="utf-8-sig")
variants = dev_summary["variant"].tolist()
metrics = ["accuracy", "macro_f1", "low_ppm_accuracy", "recall_Ethylene", "recall_Acetone", "recall_Toluene"]
labels = ["Accuracy", "Macro-F1", "≤50ppm Acc", "Ethylene Rec", "Acetone Rec", "Toluene Rec"]

fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)
x = np.arange(len(metrics))
w = 0.2
colors = ["#4C78A8", "#F28E2B", "#2e7d32", "#c62828"]
for i, variant in enumerate(variants):
    vals = dev_summary[dev_summary["variant"] == variant][metrics].values[0] * 100
    ax.bar(x + (i - 1.5) * w, vals, w, color=colors[i], edgecolor="white", label=variant, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("值 (%)")
ax.set_title("加权 MMD 开发阶段方案对比 (Batch 4-8 平均)", fontsize=13)
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "加权MMD开发阶段方案对比.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 2: Final confirm Batch 9-10 ----
final_base = final[final["variant"] == "MMD baseline"]
final_hard = final[final["variant"] == "MMD hard-gas weighted"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=180)
for ax, batch in zip(axes, [9, 10]):
    base_acc = final_base[final_base["test_batch"] == batch]["accuracy"].values[0] * 100
    hard_acc = final_hard[final_hard["test_batch"] == batch]["accuracy"].values[0] * 100
    ax.bar([0, 1], [base_acc, hard_acc], color=["#4C78A8", "#F28E2B"], edgecolor="white")
    diff = hard_acc - base_acc
    sign = "+" if diff >= 0 else ""
    ax.annotate(f"{sign}{diff:.1f}%", (1, hard_acc + 1), ha="center", fontsize=11, fontweight="bold", color="#2e7d32")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MMD 基线", "MMD 困难\n气体加权"])
    ax.set_ylabel("准确率 (%)")
    ax.set_title(f"Batch {batch}", fontsize=12)
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("加权 MMD 最终确认 (Batch 9-10)", fontsize=13, y=1.03)
fig.tight_layout()
fig.savefig(OUTPUT / "加权MMD最终确认.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 3: Batch 10 per-gas Recall ----
b10 = final[final["test_batch"] == 10]
b10_base = b10[b10["variant"] == "MMD baseline"]
b10_hard = b10[b10["variant"] == "MMD hard-gas weighted"]

fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
x = np.arange(len(GAS_NAMES))
w = 0.35
ax.bar(x - w/2, [b10_base[f"recall_{g}"].values[0] * 100 for g in GAS_NAMES], w, color="#4C78A8", edgecolor="white", label="MMD 基线")
ax.bar(x + w/2, [b10_hard[f"recall_{g}"].values[0] * 100 for g in GAS_NAMES], w, color="#F28E2B", edgecolor="white", label="MMD 困难气体加权")
for i, gas in enumerate(GAS_NAMES):
    base_r = b10_base[f"recall_{gas}"].values[0]
    hard_r = b10_hard[f"recall_{gas}"].values[0]
    diff = (hard_r - base_r) * 100
    if abs(diff) > 0.5:
        color = "#2e7d32" if diff >= 0 else "#c62828"
        y = max(base_r, hard_r) * 100 + 2
        ax.annotate(f"{diff:+.1f}%", (x[i], y), ha="center", fontsize=9, color=color, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(GAS_NAMES)
ax.set_ylabel("Recall (%)")
ax.set_title("Batch 10 逐气体 Recall：MMD 基线 vs 困难气体加权", fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "加权MMD_Batch10逐气体Recall.png", bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"Figures saved: {OUTPUT.resolve()}")
