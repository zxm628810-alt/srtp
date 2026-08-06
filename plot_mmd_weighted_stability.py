"""Plot weighted MMD stability results."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / "mmd_weighted_stability_results" / "mmd_weighted_stability_raw.csv"
OUTPUT = ROOT / "mmd_weighted_stability_results" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

raw = pd.read_csv(RESULTS, encoding="utf-8-sig")

# ---- Fig 1: Accuracy error bars ----
by_batch = raw.groupby(["variant", "test_batch"])["accuracy"].agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
colours = {"MMD baseline": "#4C78A8", "MMD hard-gas weighted": "#F28E2B"}
for variant, group in by_batch.groupby("variant"):
    ax.errorbar(group["test_batch"], group["mean"] * 100, yerr=group["std"].fillna(0) * 100,
                marker="o", capsize=4, linewidth=2, color=colours[variant], label=variant)
ax.set_xticks(range(4, 11))
ax.set_xlabel("Test Batch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Weighted MMD stability: accuracy across 3 random seeds", fontsize=12)
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT / "加权MMD稳定性误差条图.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 2: Batch 10 paired differences ----
paired = raw[raw["test_batch"] == 10].pivot(index="seed", columns="variant", values="accuracy")
paired["diff"] = (paired["MMD hard-gas weighted"] - paired["MMD baseline"]) * 100
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
colors_bar = ["#2e7d32" if d >= 0 else "#c62828" for d in paired["diff"]]
ax.bar(range(3), paired["diff"], color=colors_bar, edgecolor="white")
ax.set_xticks(range(3))
ax.set_xticklabels([f"Seed {s}" for s in paired.index])
ax.set_ylabel("Accuracy difference (weighted - baseline) [%]")
ax.set_title("Batch 10: hard-gas weighted MMD vs baseline (3 seeds)", fontsize=12)
ax.axhline(y=0, color="gray", linewidth=0.8)
for i, d in enumerate(paired["diff"]):
    ax.annotate(f"{d:+.2f}%", (i, d + 0.1 if d >= 0 else d - 0.3), ha="center", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "加权MMD稳定性Batch10成对差值.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---- Fig 3: Batch 10 per-gas Recall bars (mean across 3 seeds) ----
b10 = raw[raw["test_batch"] == 10]
fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
x = np.arange(3)
w = 0.3
gases_3 = ["Ethylene", "Acetone", "Toluene"]
for i, (gas, colour) in enumerate(zip(gases_3, ["#4C78A8", "#F28E2B", "#2e7d32"])):
    base_vals = b10[b10["variant"] == "MMD baseline"][f"recall_{gas}"].values
    hard_vals = b10[b10["variant"] == "MMD hard-gas weighted"][f"recall_{gas}"].values
    ax.bar(x[i] - w/2, base_vals.mean() * 100, w, color="#4C78A8", edgecolor="white", alpha=0.85)
    ax.bar(x[i] + w/2, hard_vals.mean() * 100, w, color="#F28E2B", edgecolor="white", alpha=0.85)
    # error bars
    ax.errorbar(x[i] - w/2, base_vals.mean() * 100, yerr=base_vals.std() * 100, fmt="none", ecolor="black", capsize=3)
    ax.errorbar(x[i] + w/2, hard_vals.mean() * 100, yerr=hard_vals.std() * 100, fmt="none", ecolor="black", capsize=3)
    diff = (hard_vals.mean() - base_vals.mean()) * 100
    c = "#2e7d32" if diff >= 0 else "#c62828"
    y = max(base_vals.mean(), hard_vals.mean()) * 100 + 3
    ax.annotate(f"{diff:+.1f}%", (x[i], y), ha="center", fontsize=10, color=c, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(gases_3)
ax.set_ylabel("Recall (%)")
ax.set_title("Batch 10 困难气体 Recall (3种子均值±标准差)", fontsize=13)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)
# Legend
from matplotlib.patches import Patch
ax.legend([Patch(color="#4C78A8"), Patch(color="#F28E2B")], ["MMD baseline", "MMD hard-gas weighted"], fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT / "加权MMD稳定性Batch10困难气体.png", bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"Figures saved: {OUTPUT.resolve()}")
