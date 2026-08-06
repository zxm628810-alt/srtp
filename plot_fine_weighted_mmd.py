"""Plot fine-grained weighted MMD results."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / "fine_weighted_mmd_results" / "fine_weighted_results.csv"
DEV_SUMMARY = ROOT / "fine_weighted_mmd_results" / "fine_weighted_dev_summary.csv"
OUTPUT = ROOT / "fine_weighted_mmd_results" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(RESULTS, encoding="utf-8-sig")
dev = df[~df["test_batch"].isin([9, 10])]
final = df[df["test_batch"].isin([9, 10])]
variants_order = ["MMD baseline", "MMD hard-gas weighted", "MMD fine-grained"]
colors = {"MMD baseline": "#4C78A8", "MMD hard-gas weighted": "#F28E2B", "MMD fine-grained": "#2e7d32"}

# ---- Fig 1: Dev phase comparison ----
dev_summary = pd.read_csv(DEV_SUMMARY, encoding="utf-8-sig")
metrics = ["accuracy", "macro_f1", "recall_Ethylene", "recall_Acetone", "recall_Toluene"]
labels = ["Accuracy", "Macro-F1", "Ethylene Rec", "Acetone Rec", "Toluene Rec"]

fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)
x = np.arange(len(metrics))
w = 0.25
for i, variant in enumerate(variants_order):
    sub = dev_summary[dev_summary["variant"] == variant]
    if sub.empty:
        continue
    vals = [sub[m].values[0] * 100 for m in metrics]
    ax.bar(x + (i - 1) * w, vals, w, color=colors[variant], edgecolor="white", label=variant, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("值 (%)")
ax.set_title("细粒度加权 MMD 开发阶段方案对比 (Batch 4-8)", fontsize=13)
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "细粒度加权MMD开发阶段对比.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 2: Final confirm Batch 9-10 ----
final_variants = [v for v in variants_order if v in final["variant"].values]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=180)
for ax, batch in zip(axes, [9, 10]):
    batch_data = final[final["test_batch"] == batch]
    accs = []
    names = []
    for v in final_variants:
        sub = batch_data[batch_data["variant"] == v]
        if not sub.empty:
            accs.append(sub["accuracy"].values[0] * 100)
            names.append(v.replace("MMD ", ""))
    ax.bar(range(len(accs)), accs, color=[colors[v] for v in final_variants], edgecolor="white")
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("准确率 (%)")
    ax.set_title(f"Batch {batch}", fontsize=12)
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("细粒度加权 MMD 最终确认 (Batch 9-10)", fontsize=13, y=1.03)
fig.tight_layout()
fig.savefig(OUTPUT / "细粒度加权MMD最终确认.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 3: Batch 10 per-gas Recall ----
b10 = final[final["test_batch"] == 10]
b10_variants = [v for v in variants_order if v in b10["variant"].values]
fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
x = np.arange(len(GAS_NAMES))
w = 0.25
for i, variant in enumerate(b10_variants):
    sub = b10[b10["variant"] == variant]
    recalls = [sub[f"recall_{g}"].values[0] * 100 for g in GAS_NAMES]
    ax.bar(x + (i - 1) * w, recalls, w, color=colors[variant], edgecolor="white", label=variant, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(GAS_NAMES)
ax.set_ylabel("Recall (%)")
ax.set_title("Batch 10 逐气体 Recall 对比", fontsize=13)
ax.legend(fontsize=8)
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "细粒度加权MMD_Batch10逐气体Recall.png", bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"Figures saved: {OUTPUT.resolve()}")
