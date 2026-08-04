"""Summarize development-only multi-task loss-weight selection."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).parent
DEV = {0.15: "multitask_weight_dev_015", 0.35: "multitask_weight_dev_035", 0.60: "multitask_weight_dev_060"}


def main() -> None:
    rows = []
    for weight, folder in DEV.items():
        table = pd.read_csv(ROOT / folder / "cross_batch_multitask_metrics.csv", encoding="utf-8-sig")
        row = {"regression_weight": weight, **table[["accuracy", "macro_f1", "mae", "r2", "low_ppm_mae"]].mean().to_dict()}
        rows.append(row)
    out = pd.DataFrame(rows)
    for metric, ascending in [("accuracy", False), ("macro_f1", False), ("mae", True), ("low_ppm_mae", True)]:
        out[f"rank_{metric}"] = out[metric].rank(ascending=ascending, method="min")
    ranks = [column for column in out if column.startswith("rank_")]
    out["mean_rank"] = out[ranks].mean(axis=1)
    out = out.sort_values("mean_rank")
    out.to_csv(ROOT / "multitask_loss_weight_selection.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
