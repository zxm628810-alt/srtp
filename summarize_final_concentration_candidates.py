"""Combine every concentration-regression candidate into one comparison CSV."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).parent
COLUMNS = ["model", "test_batch", "mae", "r2", "low_ppm_mae"]


def main() -> None:
    standard = pd.read_csv(ROOT / "concentration_drift_results" / "cross_batch_concentration_metrics.csv", encoding="utf-8-sig")[COLUMNS]
    two_stage = pd.read_csv(ROOT / "two_stage_concentration_results" / "cross_batch_two_stage_metrics.csv", encoding="utf-8-sig")[COLUMNS]
    extra = pd.read_csv(ROOT / "lowppm_residual_results" / "cross_batch_lowppm_residual_metrics.csv", encoding="utf-8-sig")[COLUMNS]
    all_metrics = pd.concat([standard, two_stage, extra], ignore_index=True)
    rows = []
    for phase, mask in [("开发阶段（Batch 4-8）", all_metrics.test_batch <= 8),
                        ("留出确认（Batch 9-10）", all_metrics.test_batch >= 9),
                        ("全部（Batch 4-10）", all_metrics.test_batch.between(4, 10))]:
        part = all_metrics[mask].groupby("model")[["mae", "r2", "low_ppm_mae"]].mean().reset_index()
        part.insert(0, "evaluation_phase", phase); rows.append(part)
    table = pd.concat(rows, ignore_index=True).sort_values(["evaluation_phase", "mae"])
    out = ROOT / "final_concentration_candidate_comparison.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
