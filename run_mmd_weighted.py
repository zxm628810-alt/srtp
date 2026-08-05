"""Select and confirm class/low-concentration weighted MMD classifiers.

All choices are made on Batches 4-8. Batches 9-10 are held out until a
single weighting strategy has been selected.
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
    DEFAULT_SEED, FEATURES, GAS_NAMES, LOW_PPM, EncoderMLP, multi_sigma_mmd, seed_everything,
)

ROOT = Path(__file__).parent
HARD_GASES = {"Ethylene", "Acetone", "Toluene"}
MMD_LAMBDA = 0.5  # locked by the earlier Batch 4-8 MMD selection experiment


def train_weighted(x: np.ndarray, y: np.ndarray, batch_ids: np.ndarray, ppm: np.ndarray,
                   class_weight: float, low_weight: float, seed: int) -> EncoderMLP:
    seed_everything(seed)
    indices = np.arange(len(x))
    train_idx, valid_idx = train_test_split(indices, test_size=.15, random_state=seed, stratify=y)
    sample_weight = np.ones(len(x), dtype=np.float32)
    hard_ids = [GAS_NAMES.index(g) for g in HARD_GASES]
    sample_weight[np.isin(y, hard_ids)] *= class_weight
    sample_weight[ppm <= LOW_PPM] *= low_weight

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
            if src.sum() >= 2 and tgt.sum() >= 2:
                loss = cls_loss + MMD_LAMBDA * multi_sigma_mmd(z[src], z[tgt])
            else:
                loss = cls_loss
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.CrossEntropyLoss()(model(xv), yv).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, no_improve = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= 25:
                break
    model.load_state_dict(best_state)
    return model


def evaluate(train: pd.DataFrame, test: pd.DataFrame, name: str,
             class_weight: float, low_weight: float, seed: int) -> dict:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(train[FEATURES].values)
    xte = scaler.transform(test[FEATURES].values)
    ytr = train.gas_name.map({g: i for i, g in enumerate(GAS_NAMES)}).values
    model = train_weighted(xtr, ytr, train.batch_id.values, train.concentration_ppm.values,
                           class_weight, low_weight, seed)
    with torch.no_grad():
        pred = np.array(GAS_NAMES)[model(torch.tensor(xte, dtype=torch.float32)).argmax(1).numpy()]
    low = test.concentration_ppm.values <= LOW_PPM
    recalls = recall_score(test.gas_name, pred, labels=GAS_NAMES, average=None, zero_division=0)
    row = {
        "variant": name, "class_weight": class_weight, "low_ppm_weight": low_weight,
        "test_batch": int(test.batch_id.iloc[0]), "seed": seed,
        "accuracy": accuracy_score(test.gas_name, pred),
        "macro_f1": f1_score(test.gas_name, pred, labels=GAS_NAMES, average="macro", zero_division=0),
        "low_ppm_accuracy": accuracy_score(test.gas_name[low], pred[low]) if low.any() else np.nan,
    }
    row.update({f"recall_{g}": v for g, v in zip(GAS_NAMES, recalls)})
    return row


def rank_variants(rows: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    metrics = ["accuracy", "macro_f1", "low_ppm_accuracy", "recall_Ethylene", "recall_Acetone", "recall_Toluene"]
    summary = rows.groupby("variant")[metrics].mean().reset_index()
    ranks = summary[metrics].rank(axis=0, ascending=False, method="average")
    summary["mean_rank"] = ranks.mean(axis=1)
    return summary.loc[summary.mean_rank.idxmin(), "variant"], summary.sort_values("mean_rank")


def main(csv_path: Path, output: Path, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    variants = {
        "MMD baseline": (1.0, 1.0),
        "MMD hard-gas weighted": (1.5, 1.0),
        "MMD low-ppm weighted": (1.0, 1.5),
        "MMD combined weighted": (1.5, 1.5),
    }
    dev_rows = []
    for name, (cw, lw) in variants.items():
        print(f"\n--- DEV {name} ---")
        for b in range(4, 9):
            row = evaluate(df[df.batch_id < b], df[df.batch_id == b], name, cw, lw, seed)
            dev_rows.append(row)
            print(f"Batch {b}: Acc={row['accuracy']:.4f}, F1={row['macro_f1']:.4f}")
    dev = pd.DataFrame(dev_rows)
    selected, summary = rank_variants(dev)
    summary.to_csv(output / "dev_selection_summary.csv", index=False, encoding="utf-8-sig")
    print("\nSelected variant:", selected)

    cw, lw = variants[selected]
    final_rows = []
    for name, weights in [("MMD baseline", variants["MMD baseline"]), (selected, (cw, lw))]:
        print(f"\n--- FINAL {name} ---")
        for b in [9, 10]:
            row = evaluate(df[df.batch_id < b], df[df.batch_id == b], name, weights[0], weights[1], seed)
            final_rows.append(row)
            print(f"Batch {b}: Acc={row['accuracy']:.4f}, F1={row['macro_f1']:.4f}")
    all_rows = pd.concat([dev.assign(phase="dev"), pd.DataFrame(final_rows).assign(phase="final")])
    all_rows.to_csv(output / "mmd_weighted_results.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "mmd_weighted_results")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    main(args.csv, args.output, args.seed)
