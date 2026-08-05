"""Plot Fusion MMD results."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / "fusion_mmd_results" / "fusion_mmd_results.csv"
OUTPUT = ROOT / "fusion_mmd_results" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(RESULTS, encoding="utf-8-sig")

# Deduplicate: keep only the first occurrence of each (model, mmd_lambda, test_batch) combo
df = df.drop_duplicates(subset=["model", "mmd_lambda", "test_batch"], keep="first")

# ---- Fig 1: Dev phase comparison (acc + mae side by side) ----
dev = df[df["test_batch"].isin([4, 5, 6, 7, 8])]
lambdas = sorted(dev["mmd_lambda"].unique())
batches_dev = [4, 5, 6, 7, 8]
colors = ["#4C78A8", "#F28E2B", "#c62828"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
for ax, metric, ylabel in zip(axes, ["accuracy", "mae"], ["准确率 (%)", "MAE (ppm)"]):
    x = np.arange(len(batches_dev))
    for i, lam in enumerate(lambdas):
        sub = dev[dev["mmd_lambda"] == lam]
        label = f"基线 (无MMD)" if lam == 0 else f"λ_mmd={lam}"
        ls = "-" if lam == 0 else "--"
        vals = sub[metric].values * 100 if metric == "accuracy" else sub[metric].values
        ax.plot(x, vals, "o" + ls, color=colors[i], linewidth=2, markersize=7, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Batch {b}" for b in batches_dev])
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Fusion MMD 开发阶段对比 (Batch 4-8)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT / "FusionMMD开发阶段对比.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 2: Final confirm (Batch 9-10) ----
final = df[df["test_batch"].isin([9, 10])]
final_base = final[final["mmd_lambda"] == 0.0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=180)
for ax, metric, ylabel in zip(axes, ["accuracy", "mae"], ["准确率 (%)", "MAE (ppm)"]):
    x = np.arange(2)
    vals = final_base[metric].values * 100 if metric == "accuracy" else final_base[metric].values
    ax.bar(x, vals, color="#4C78A8", edgecolor="white")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}", (x[i], v + 1), ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["Batch 9", "Batch 10"])
    ax.set_ylabel(ylabel)
    ax.set_title("Fusion 基线 (无MMD)" if metric == "accuracy" else "")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Fusion MMD 最终确认 (Batch 9-10，MMD 被淘汰，仅报告基线)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT / "FusionMMD最终确认.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 3: Batch 7 breakdown (where MMD failed most) ----
b7 = df[df["test_batch"] == 7]
x = np.arange(len(GAS_NAMES))
w = 0.25

fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
for i, (lam, color, label) in enumerate(zip([0.0, 0.1, 0.5], colors, ["基线", "λ=0.1", "λ=0.5"])):
    sub = b7[b7["mmd_lambda"] == lam]
    recalls = [sub[f"recall_{g}"].values[0] * 100 for g in GAS_NAMES]
    ax.bar(x + (i - 1) * w, recalls, w, color=color, edgecolor="white", label=label, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(GAS_NAMES)
ax.set_ylabel("Recall (%)")
ax.set_title("Batch 7 逐气体 Recall：Fusion MMD 各 λ 对比（MMD 在此批次失效最严重）", fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "FusionMMD_Batch7崩溃分析.png", bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"Figures saved: {OUTPUT.resolve()}")
