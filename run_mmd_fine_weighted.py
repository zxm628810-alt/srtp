"""Fine-grained per-gas weighting on top of MMD (lambda=0.5, locked).

Previous experiments:
  - Plain MMD: Batch 10 +1.5~1.6% (stable across 3 seeds)
  - Hard-gas weighted MMD (Eth/Ace/Tol all 1.5): unstable across seeds
    Ethylene improves but Toluene/Acetone are sacrificed.

Hypothesis: the three "hard gases" have different failure mechanisms.
  - Ethylene: confused with Toluene → extra weight should help
  - Acetone: MMD already gives +7% → only needs a small boost (1.2)
  - Toluene: too few training samples, not an attention problem → keep at 1.0

This experiment compares three variants:
  1. MMD baseline      — all weights 1.0
  2. MMD hard-gas       — Eth=1.5, Ace=1.5, Tol=1.5  (old, for reference)
  3. MMD fine-grained   — Eth=1.5, Ace=1.2, Tol=1.0  (new)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_domain_adaptation import (
    DEFAULT_SEED, FEATURES, GAS_NAMES, LOW_PPM, EncoderMLP,
    multi_sigma_mmd, seed_everything,
)

ROOT = Path(__file__).parent
MMD_LAMBDA = 0.5

# Per-gas weight dictionaries  (key = gas_name → multiplier)
WEIGHT_PRESETS = {
    "MMD baseline":           {"Ethylene": 1.0, "Acetone": 1.0, "Toluene": 1.0},
    "MMD hard-gas weighted":  {"Ethylene": 1.5, "Acetone": 1.5, "Toluene": 1.5},
    "MMD fine-grained":       {"Ethylene": 1.5, "Acetone": 1.2, "Toluene": 1.0},
}


def train_fine_weighted(
    x: np.ndarray, y: np.ndarray, batch_ids: np.ndarray,
    gas_weights: dict[str, float], seed: int,
) -> EncoderMLP:
    """Train MMD encoder with per-sample weights from per-gas multipliers."""
    seed_everything(seed)
    indices = np.arange(len(x))
    train_idx, valid_idx = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=y,
    )
    y_names = np.array(GAS_NAMES)[y]

    # Build per-sample weights from per-gas multipliers
    sample_weight = np.ones(len(x), dtype=np.float32)
    for gas, mult in gas_weights.items():
        sample_weight[y_names == gas] *= mult

    model = EncoderMLP(x.shape[1], len(GAS_NAMES))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction="none")

    loader = DataLoader(
        TensorDataset(
            torch.tensor(x[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.long),
            torch.tensor(batch_ids[train_idx], dtype=torch.long),
            torch.tensor(sample_weight[train_idx], dtype=torch.float32),
        ), batch_size=256, shuffle=True,
    )
    xv = torch.tensor(x[valid_idx], dtype=torch.float32)
    yv = torch.tensor(y[valid_idx], dtype=torch.long)

    best_state, best_loss, no_improve = None, float("inf"), 0
    for _ in range(250):
        model.train()
        for xb, yb, bb, wb in loader:
            opt.zero_grad()
            logits, z = model(xb, return_encoded=True)
            cls_loss = (ce(logits, yb) * wb).sum() / wb.sum()
            median_batch = bb.median()
            src, tgt = bb <= median_batch, bb > median_batch
            loss = cls_loss
            if src.sum() >= 2 and tgt.sum() >= 2:
                loss = loss + MMD_LAMBDA * multi_sigma_mmd(z[src], z[tgt])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.CrossEntropyLoss()(model(xv), yv).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 25:
                break

    model.load_state_dict(best_state)
    return model


def evaluate(
    train: pd.DataFrame, test: pd.DataFrame,
    variant_name: str, gas_weights: dict[str, float], seed: int,
) -> dict:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(train[FEATURES].values)
    xte = scaler.transform(test[FEATURES].values)
    ytr = train.gas_name.map({g: i for i, g in enumerate(GAS_NAMES)}).values

    model = train_fine_weighted(xtr, ytr, train.batch_id.values, gas_weights, seed)

    with torch.no_grad():
        pred = np.array(GAS_NAMES)[
            model(torch.tensor(xte, dtype=torch.float32)).argmax(1).numpy()
        ]

    low = test.concentration_ppm.values <= LOW_PPM
    recalls = recall_score(test.gas_name, pred, labels=GAS_NAMES, average=None, zero_division=0)
    row = {
        "variant": variant_name, "test_batch": int(test.batch_id.iloc[0]),
        "seed": seed,
        "accuracy": accuracy_score(test.gas_name, pred),
        "macro_f1": f1_score(test.gas_name, pred, labels=GAS_NAMES, average="macro", zero_division=0),
        "low_ppm_accuracy": accuracy_score(test.gas_name[low], pred[low]) if low.any() else np.nan,
    }
    row.update({f"recall_{g}": v for g, v in zip(GAS_NAMES, recalls)})
    return row


def rank_variants(df: pd.DataFrame) -> str:
    metrics = ["accuracy", "macro_f1", "recall_Ethylene", "recall_Acetone", "recall_Toluene"]
    summary = df.groupby("variant")[metrics].mean().reset_index()
    ranks = summary[metrics].rank(ascending=False, method="average")
    summary["mean_rank"] = ranks.mean(axis=1)
    best = summary.loc[summary["mean_rank"].idxmin(), "variant"]
    return best, summary


def main(csv_path: Path, output: Path, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(csv_path, encoding="utf-8-sig")

    print("=" * 70)
    print("FINE-GRAINED WEIGHTED MMD")
    print(f"Comparing: {list(WEIGHT_PRESETS.keys())}")
    print(f"MMD lambda = {MMD_LAMBDA} (locked)")
    print("=" * 70)

    # ===== DEV: batches 4-8, select best variant =====
    print("\n--- DEV PHASE (batches 4-8) ---")
    dev_rows = []
    for name, weights in WEIGHT_PRESETS.items():
        print(f"\n{name}: {weights}")
        for b in range(4, 9):
            row = evaluate(
                df_all[df_all["batch_id"] < b], df_all[df_all["batch_id"] == b],
                name, weights, seed,
            )
            dev_rows.append(row)
            print(f"  Batch {b}: Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}  "
                  f"Ethyl={row['recall_Ethylene']:.3f}  "
                  f"Acet={row['recall_Acetone']:.3f}  "
                  f"Tol={row['recall_Toluene']:.3f}")

    dev = pd.DataFrame(dev_rows)
    selected, dev_summary = rank_variants(dev)
    dev_summary.to_csv(output / "fine_weighted_dev_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n--- DEV SUMMARY ---")
    print(dev_summary.to_string(index=False))
    print(f"\nSelected: {selected}")

    # ===== FINAL: batches 9-10, confirm vs baseline =====
    print(f"\n{'=' * 70}")
    print(f"FINAL PHASE (batches 9-10)")
    print("=" * 70)

    final_rows = []
    # Always run baseline for reference
    for variant in ["MMD baseline", selected]:
        weights = WEIGHT_PRESETS[variant]
        print(f"\n--- {variant} ---")
        for b in [9, 10]:
            row = evaluate(
                df_all[df_all["batch_id"] < b], df_all[df_all["batch_id"] == b],
                variant, weights, seed,
            )
            final_rows.append(row)
            print(f"  Batch {b}: Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}  "
                  f"Ethyl={row['recall_Ethylene']:.3f}  "
                  f"Acet={row['recall_Acetone']:.3f}  "
                  f"Tol={row['recall_Toluene']:.3f}")

    # ===== Save all =====
    all_rows = (dev_rows + final_rows)
    pd.DataFrame(all_rows).to_csv(output / "fine_weighted_results.csv", index=False, encoding="utf-8-sig")

    # ===== Print final comparison =====
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON (batches 9-10)")
    print("=" * 70)
    for b in [9, 10]:
        base = [r for r in final_rows if r["variant"] == "MMD baseline" and r["test_batch"] == b][0]
        best = [r for r in final_rows if r["variant"] == selected and r["test_batch"] == b][0]
        acc_diff = (best["accuracy"] - base["accuracy"]) * 100
        eth_diff = (best["recall_Ethylene"] - base["recall_Ethylene"]) * 100
        ace_diff = (best["recall_Acetone"] - base["recall_Acetone"]) * 100
        tol_diff = (best["recall_Toluene"] - base["recall_Toluene"]) * 100
        print(f"\nBatch {b}: {selected} vs MMD baseline")
        print(f"  Accuracy: {base['accuracy']:.4f} → {best['accuracy']:.4f} "
              f"({'+' if acc_diff >= 0 else ''}{acc_diff:.1f}%)")
        print(f"  Ethylene: {base['recall_Ethylene']:.3f} → {best['recall_Ethylene']:.3f} "
              f"({'+' if eth_diff >= 0 else ''}{eth_diff:.1f}%)")
        print(f"  Acetone:  {base['recall_Acetone']:.3f} → {best['recall_Acetone']:.3f} "
              f"({'+' if ace_diff >= 0 else ''}{ace_diff:.1f}%)")
        print(f"  Toluene:  {base['recall_Toluene']:.3f} → {best['recall_Toluene']:.3f} "
              f"({'+' if tol_diff >= 0 else ''}{tol_diff:.1f}%)")

    print(f"\nResults: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "fine_weighted_mmd_results")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    main(args.csv, args.output, args.seed)
