"""Descriptive PCA visualizations for sensor drift across UCI gas batches.

PCA is fitted to standardized sensor readings from all batches only for visual
inspection. It is not used as a predictive-model test or a drift-correction
feature, so this descriptive fit does not alter the strict rolling experiments.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).parent
DATA = ROOT / "all_batches.csv"
OUTPUT = ROOT / "figures"
FEATURES = [f"feature_{i}" for i in range(1, 129)]
GASES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
GAS_COLORS = dict(zip(GASES, plt.get_cmap("tab10").colors[:6]))
BATCH_COLORS = plt.get_cmap("tab10").colors


def sampled(frame: pd.DataFrame, group_columns: list[str], limit: int = 300) -> pd.DataFrame:
    """Deterministically cap dense groups so scatter points remain readable."""
    groups = [group.sample(min(len(group), limit), random_state=42)
              for _, group in frame.groupby(group_columns)]
    return pd.concat(groups, ignore_index=True)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA, encoding="utf-8-sig")
    scaled = StandardScaler().fit_transform(raw[FEATURES])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    df = raw[["batch_id", "gas_name", "concentration_ppm"]].copy()
    df[["PC1", "PC2"]] = coords
    df.to_csv(
        OUTPUT / "pca_coordinates.csv", index=False, encoding="utf-8-sig"
    )

    # 1. Overall movement by batch.
    batch_points = sampled(df, ["batch_id"], limit=500)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    for batch in range(1, 11):
        group = batch_points[batch_points.batch_id == batch]
        ax.scatter(group.PC1, group.PC2, s=9, alpha=.38, color=BATCH_COLORS[batch - 1], label=f"Batch {batch}")
    ax.set(title="PCA of standardized sensor features: batches 1-10", xlabel="PC1", ylabel="PC2")
    ax.legend(ncol=2, markerscale=1.8, fontsize=8)
    ax.grid(alpha=.22)
    fig.tight_layout()
    fig.savefig(OUTPUT / "pca_by_batch.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Same PCA space, compare gas distributions early vs. late.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), dpi=160, sharex=True, sharey=True)
    for ax, batches, subtitle in zip(axes, [[1, 2, 3], [8, 9, 10]], ["Early batches (1-3)", "Later batches (8-10)"]):
        part = sampled(df[df.batch_id.isin(batches)], ["gas_name", "batch_id"], limit=230)
        for gas in GASES:
            group = part[part.gas_name == gas]
            ax.scatter(group.PC1, group.PC2, s=9, alpha=.42, color=GAS_COLORS[gas], label=gas)
        ax.set_title(subtitle)
        ax.set_xlabel("PC1")
        ax.grid(alpha=.22)
    axes[0].set_ylabel("PC2")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.02), fontsize=8)
    fig.suptitle("PCA gas-class distributions: early versus later batches", y=.99)
    fig.tight_layout()
    fig.savefig(OUTPUT / "pca_early_vs_late_by_gas.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Class-centroid trajectories make each gas's drift direction visible.
    centroids = df.groupby(["gas_name", "batch_id"], as_index=False)[["PC1", "PC2"]].mean()
    centroids.to_csv(OUTPUT / "pca_gas_centroids.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    for gas in GASES:
        group = centroids[centroids.gas_name == gas].sort_values("batch_id")
        ax.plot(group.PC1, group.PC2, "-o", linewidth=2, markersize=4, color=GAS_COLORS[gas], label=gas)
        if not group.empty:
            first, last = group.iloc[0], group.iloc[-1]
            ax.annotate(f"{gas} B{int(first.batch_id)}", (first.PC1, first.PC2), xytext=(4, 4), textcoords="offset points", fontsize=7)
            ax.annotate(f"B{int(last.batch_id)}", (last.PC1, last.PC2), xytext=(4, -9), textcoords="offset points", fontsize=7)
    ax.set(title="PCA class-centroid trajectories across batches", xlabel="PC1", ylabel="PC2")
    ax.grid(alpha=.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "pca_gas_centroid_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    explained = pca.explained_variance_ratio_
    print(f"PC1 explained variance: {explained[0]:.2%}")
    print(f"PC2 explained variance: {explained[1]:.2%}")
    print(f"PCA figures saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
