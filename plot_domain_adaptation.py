"""Plot domain adaptation results: dev selection + final comparison + per-gas recall."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / "domain_adaptation_results" / "domain_adaptation_results.csv"
OUTPUT = ROOT / "domain_adaptation_results" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(RESULTS, encoding="utf-8-sig")

# ---- Fig 1: Dev phase λ selection ----
dev = df[df["test_batch"].isin([4, 5, 6, 7, 8])]
lambdas = sorted(dev["mmd_lambda"].unique())
batches_dev = [4, 5, 6, 7, 8]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
colors = ["#4C78A8", "#F28E2B", "#2e7d32", "#c62828"]

for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
    x = np.arange(len(batches_dev))
    for i, lam in enumerate(lambdas):
        sub = dev[dev["mmd_lambda"] == lam]
        label = f"基线 (无MMD)" if lam == 0 else f"λ={lam}"
        ls = "-" if lam == 0 else "--"
        ax.plot(x, sub[metric].values * 100, "o" + ls, color=colors[i], linewidth=2,
                markersize=7, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Batch {b}" for b in batches_dev])
    ylabel = "准确率 (%)" if metric == "accuracy" else "Macro-F1 (%)"
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if metric == "accuracy":
        ax.set_ylim(70, 105)

fig.suptitle("领域自适应开发阶段 λ 选择 (Batch 4-8)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT / "领域自适应开发阶段λ选择.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 2: Final confirm bar chart (Batch 9-10) ----
final = df[df["test_batch"].isin([9, 10])]
final_base = final[final["mmd_lambda"] == 0.0]
final_mmd = final[final["mmd_lambda"] == 0.5]

fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
x = np.arange(2)
w = 0.3
ax.bar(x - w/2, final_base["accuracy"].values * 100, w, color="#4C78A8", edgecolor="white", label="基线 (无MMD)")
ax.bar(x + w/2, final_mmd["accuracy"].values * 100, w, color="#F28E2B", edgecolor="white", label="MMD λ=0.5")
for i, (base_acc, mmd_acc) in enumerate(zip(final_base["accuracy"].values, final_mmd["accuracy"].values)):
    diff = (mmd_acc - base_acc) * 100
    color = "#2e7d32" if diff >= 0 else "#c62828"
    y = max(base_acc, mmd_acc) * 100 + 1.5
    ax.annotate(f"{diff:+.1f}%", (x[i], y), ha="center", fontsize=11, color=color, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Batch 9", "Batch 10"])
ax.set_ylabel("准确率 (%)")
ax.set_title("领域自适应最终确认 (λ=0.5)", fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(0, 90)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "领域自适应最终确认.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 3: Per-gas Recall at Batch 10 ----
b10 = df[df["test_batch"] == 10]
b10_base = b10[b10["mmd_lambda"] == 0.0]
b10_mmd = b10[b10["mmd_lambda"] == 0.5]

fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
x = np.arange(len(GAS_NAMES))
w = 0.3
bars1 = ax.bar(x - w/2, [b10_base[f"recall_{g}"].values[0] * 100 for g in GAS_NAMES], w, color="#4C78A8", edgecolor="white", label="基线 (无MMD)")
bars2 = ax.bar(x + w/2, [b10_mmd[f"recall_{g}"].values[0] * 100 for g in GAS_NAMES], w, color="#F28E2B", edgecolor="white", label="MMD λ=0.5")
for i, gas in enumerate(GAS_NAMES):
    base_r = b10_base[f"recall_{gas}"].values[0]
    mmd_r = b10_mmd[f"recall_{gas}"].values[0]
    diff = (mmd_r - base_r) * 100
    if abs(diff) > 0.5:
        color = "#2e7d32" if diff >= 0 else "#c62828"
        y = max(base_r, mmd_r) * 100 + 2
        ax.annotate(f"{diff:+.1f}%", (x[i], y), ha="center", fontsize=9, color=color, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(GAS_NAMES)
ax.set_ylabel("Recall (%)")
ax.set_title("Batch 10 逐气体 Recall：基线 vs MMD 领域自适应", fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "领域自适应Batch10逐气体Recall.png", bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"Figures saved: {OUTPUT.resolve()}")
