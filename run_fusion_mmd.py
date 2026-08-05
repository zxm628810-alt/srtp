"""MMD domain adaptation × Fusion history-calibrated multi-task DNN.

Combines the two strongest methods discovered so far:
  1. Fusion dual-branch architecture (sensor + history calibration)
     — best for concentration prediction (low-ppm MAE 10.90)
  2. MMD domain adaptation (λ=0.5)
     — best for long-term classification (+3.8% on Batch 10)

Key design:
  - Sensor branch (128→256→128) and history branch (128→128→64) each have
    their own MMD regularisation applied to their output representations
  - Both branches are forced to be "batch-invariant" independently
  - The fused representation then does classification + regression
  - Total loss = CE + λ_reg * SmoothL1 + λ_mmd * (MMD_sensor + MMD_history)

The model NEVER receives the current batch ID — only historical calibration
features computed from earlier batches.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_drift_experiments import FEATURES, GAS_NAMES, select_x

ROOT = Path(__file__).parent
LOW_PPM = 50.0
SEED = 42
LAMBDA_REG = 0.60   # optimal from multi-task weight selection experiment


# ---- MMD utilities (same as run_domain_adaptation.py) ----
def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    x_norm = (x ** 2).sum(1).unsqueeze(1)
    y_norm = (y ** 2).sum(1).unsqueeze(0)
    dist = x_norm + y_norm - 2.0 * x @ y.T
    return torch.exp(-dist / (2.0 * sigma ** 2))


def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    k_xx = gaussian_kernel(x, y, sigma)
    k_yy = gaussian_kernel(y, y, sigma)
    k_xy = gaussian_kernel(x, y, sigma)
    n, m = x.size(0), y.size(0)
    k_xx_val = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1)) if n > 1 else k_xx.mean()
    k_yy_val = (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1)) if m > 1 else k_yy.mean()
    k_xy_val = k_xy.mean()
    return k_xx_val + k_yy_val - 2 * k_xy_val


def multi_sigma_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        combined = torch.cat([x, y], dim=0)
        dists = torch.cdist(combined, combined)
        median_dist = dists[dists > 0].median()
    sigmas = [median_dist * s for s in [0.5, 1.0, 2.0]]
    return sum(mmd_loss(x, y, sigma=s) for s in sigmas) / len(sigmas)


# ---- Fusion MMD Model ----
class FusionMMDNet(nn.Module):
    """Fusion model with MMD applied to both branches independently."""

    def __init__(self, n_classes: int = len(GAS_NAMES)):
        super().__init__()
        # Sensor branch: 128 → 256 → 128
        self.sensor_encoder = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
        )
        # History branch: 128 → 128 → 64
        self.history_encoder = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.ReLU(),
        )
        # Fusion head: 128+64=192 → 128 → 64
        self.fusion = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.classifier = nn.Linear(64, n_classes)
        self.regressor = nn.Linear(64, 1)

    def forward(self, sensor: torch.Tensor, history: torch.Tensor,
                return_encoded: bool = False):
        s_encoded = self.sensor_encoder(sensor)    # (N, 128)
        h_encoded = self.history_encoder(history)   # (N, 64)
        fused = self.fusion(torch.cat([s_encoded, h_encoded], dim=1))
        logits = self.classifier(fused)
        reg = self.regressor(fused).squeeze(1)
        if return_encoded:
            return logits, reg, s_encoded, h_encoded
        return logits, reg


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def build_features(frame: pd.DataFrame, history: pd.DataFrame):
    """Extract sensor and history-calibration features (no batch ID)."""
    all_x = select_x(frame, "dnn_sensor_history_baseline", history=history, history_window=3)
    sensor = all_x[FEATURES]
    hist = all_x[[f"history_relative_{name}" for name in FEATURES]]
    return sensor, hist


def train_model(sensor_train: np.ndarray, history_train: np.ndarray,
                y_class: np.ndarray, y_ppm: np.ndarray,
                batch_ids: np.ndarray,
                mmd_lambda: float, lr: float = 1e-3,
                epochs: int = 250, patience: int = 25) -> tuple[FusionMMDNet, StandardScaler]:
    """Train Fusion MMD model."""

    seed_everything()
    ppm_scaler = StandardScaler()
    y_reg = ppm_scaler.fit_transform(y_ppm.reshape(-1, 1)).ravel()

    train_idx, valid_idx = train_test_split(
        np.arange(len(sensor_train)), test_size=0.15, random_state=SEED, stratify=y_class,
    )

    model = FusionMMDNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    huber = nn.SmoothL1Loss()

    # Training tensors
    s_tr = torch.tensor(sensor_train[train_idx], dtype=torch.float32)
    h_tr = torch.tensor(history_train[train_idx], dtype=torch.float32)
    c_tr = torch.tensor(y_class[train_idx], dtype=torch.long)
    r_tr = torch.tensor(y_reg[train_idx], dtype=torch.float32)
    b_tr = torch.tensor(batch_ids[train_idx], dtype=torch.long)

    # Validation tensors
    s_val = torch.tensor(sensor_train[valid_idx], dtype=torch.float32)
    h_val = torch.tensor(history_train[valid_idx], dtype=torch.float32)
    c_val = torch.tensor(y_class[valid_idx], dtype=torch.long)
    r_val = torch.tensor(y_reg[valid_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(s_tr, h_tr, c_tr, r_tr, b_tr),
        batch_size=256, shuffle=True,
    )

    best_state, best_loss, no_improve = None, float("inf"), 0

    for _ in range(epochs):
        model.train()
        for sb, hb, cb, rb, bb in train_loader:
            opt.zero_grad()

            if mmd_lambda > 0:
                logits, reg, s_enc, h_enc = model(sb, hb, return_encoded=True)
                cls_loss = ce(logits, cb)
                reg_loss = huber(reg, rb)

                # Split by batch median → source (old) vs target (recent)
                median_b = bb.median()
                src = bb <= median_b
                tgt = bb > median_b

                mmd_term = 0.0
                if src.sum() >= 2 and tgt.sum() >= 2:
                    mmd_term = multi_sigma_mmd(s_enc[src], s_enc[tgt]) \
                             + multi_sigma_mmd(h_enc[src], h_enc[tgt])

                loss = cls_loss + LAMBDA_REG * reg_loss + mmd_lambda * mmd_term
            else:
                logits, reg = model(sb, hb)
                cls_loss = ce(logits, cb)
                reg_loss = huber(reg, rb)
                loss = cls_loss + LAMBDA_REG * reg_loss

            loss.backward()
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits, reg = model(s_val, h_val)
            val_loss = (ce(logits, c_val) + LAMBDA_REG * huber(reg, r_val)).item()

        if val_loss < best_loss - 1e-5:
            best_loss, best_state = val_loss, {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, ppm_scaler


def evaluate(train: pd.DataFrame, test: pd.DataFrame, mmd_lambda: float,
             output: Path) -> dict:
    """Train on batches < k, test on batch k."""

    # Build features (history calibration uses training data only)
    s_train, h_train = build_features(train, train)
    s_test, h_test = build_features(test, train)

    # Scale
    s_scaler = StandardScaler()
    h_scaler = StandardScaler()
    s_train_s = s_scaler.fit_transform(s_train)
    h_train_s = h_scaler.fit_transform(h_train)
    s_test_s = s_scaler.transform(s_test)
    h_test_s = h_scaler.transform(h_test)

    class_to_id = {g: i for i, g in enumerate(GAS_NAMES)}
    y_class = train.gas_name.map(class_to_id).values
    batch_ids = train.batch_id.values

    model, ppm_scaler = train_model(
        s_train_s, h_train_s, y_class,
        train.concentration_ppm.values, batch_ids, mmd_lambda,
    )

    model.eval()
    with torch.no_grad():
        logits, reg = model(
            torch.tensor(s_test_s, dtype=torch.float32),
            torch.tensor(h_test_s, dtype=torch.float32),
        )
        pred_ids = logits.argmax(dim=1).numpy()
        pred_ppm = np.maximum(
            ppm_scaler.inverse_transform(reg.numpy().reshape(-1, 1)).ravel(), 0,
        )

    pred_gas = np.array(GAS_NAMES)[pred_ids]
    batch = int(test.batch_id.iloc[0])
    low = test.concentration_ppm <= LOW_PPM

    row = {
        "model": "fusion_mmd" if mmd_lambda > 0 else "fusion_baseline",
        "mmd_lambda": mmd_lambda,
        "test_batch": batch,
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": accuracy_score(test.gas_name, pred_gas),
        "macro_f1": f1_score(test.gas_name, pred_gas, labels=GAS_NAMES, average="macro", zero_division=0),
        "mae": mean_absolute_error(test.concentration_ppm, pred_ppm),
        "r2": r2_score(test.concentration_ppm, pred_ppm),
    }

    recalls = recall_score(test.gas_name, pred_gas, labels=GAS_NAMES, average=None, zero_division=0)
    row.update({f"recall_{g}": float(v) for g, v in zip(GAS_NAMES, recalls)})
    row["low_ppm_n"] = int(low.sum())
    row["low_ppm_mae"] = mean_absolute_error(
        test.loc[low, "concentration_ppm"], pred_ppm[low],
    ) if low.any() else np.nan

    return row


def main(csv_path: Path, output: Path, dev_batches: list[int],
         final_batches: list[int], mmd_lambdas: list[float]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # ===== Dev phase: select λ_mmd on batches 4-8 =====
    print("=" * 70)
    print("DEV PHASE: Fusion MMD — select λ_mmd on batches 4-8")
    print(f"(λ_reg = {LAMBDA_REG} fixed from multi-task weight selection)")
    print("=" * 70)

    dev_rows = []
    for lam in mmd_lambdas:
        label = f"MMD λ={lam}" if lam > 0 else "Baseline (no MMD)"
        print(f"\n--- {label} ---")
        for batch in dev_batches:
            train = df[df["batch_id"] < batch]
            test = df[df["batch_id"] == batch]
            row = evaluate(train, test, lam, output)
            dev_rows.append(row)
            print(f"  Batch {batch}: Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}  "
                  f"MAE={row['mae']:.1f}  ≤50ppm MAE={row['low_ppm_mae']:.1f}")

    dev_results = pd.DataFrame(dev_rows)
    dev_summary = dev_results.groupby("mmd_lambda").agg(
        avg_accuracy=("accuracy", "mean"),
        avg_f1=("macro_f1", "mean"),
        avg_mae=("mae", "mean"),
        avg_low_ppm_mae=("low_ppm_mae", "mean"),
    ).reset_index()

    for col in ["avg_accuracy", "avg_f1"]:
        dev_summary[f"rank_{col}"] = dev_summary[col].rank(ascending=False)
    for col in ["avg_mae", "avg_low_ppm_mae"]:
        dev_summary[f"rank_{col}"] = dev_summary[col].rank(ascending=True)
    dev_summary["mean_rank"] = (
        dev_summary["rank_avg_accuracy"] + dev_summary["rank_avg_f1"] +
        dev_summary["rank_avg_mae"] + dev_summary["rank_avg_low_ppm_mae"]
    ) / 4

    print(f"\n{'=' * 70}")
    print("DEV SUMMARY")
    print("=" * 70)
    print(dev_summary[["mmd_lambda", "avg_accuracy", "avg_f1", "avg_mae",
                        "avg_low_ppm_mae", "mean_rank"]].to_string(index=False))

    best_lambda = dev_summary.loc[dev_summary["mean_rank"].idxmin(), "mmd_lambda"]
    print(f"\nSelected λ_mmd = {best_lambda}")

    # ===== Final confirm on batches 9-10 =====
    print(f"\n{'=' * 70}")
    print(f"FINAL PHASE: Confirm λ_mmd={best_lambda} on batches 9-10")
    print("=" * 70)

    final_rows = []
    print("\n--- Baseline (no MMD) ---")
    for batch in final_batches:
        train = df[df["batch_id"] < batch]
        test = df[df["batch_id"] == batch]
        row = evaluate(train, test, mmd_lambda=0.0, output=output)
        final_rows.append(row)
        print(f"  Batch {batch}: Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}  "
              f"MAE={row['mae']:.1f}  ≤50ppm MAE={row['low_ppm_mae']:.1f}")

    print(f"\n--- MMD λ_mmd={best_lambda} ---")
    for batch in final_batches:
        train = df[df["batch_id"] < batch]
        test = df[df["batch_id"] == batch]
        row = evaluate(train, test, mmd_lambda=best_lambda, output=output)
        final_rows.append(row)
        print(f"  Batch {batch}: Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}  "
              f"MAE={row['mae']:.1f}  ≤50ppm MAE={row['low_ppm_mae']:.1f}")

    # ===== Save =====
    all_results = pd.concat([dev_results, pd.DataFrame(final_rows)], ignore_index=True)
    all_results.to_csv(output / "fusion_mmd_results.csv", index=False, encoding="utf-8-sig")
    print(f"\nResults saved: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "fusion_mmd_results")
    parser.add_argument("--dev-batches", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--final-batches", nargs="+", type=int, default=[9, 10])
    parser.add_argument("--mmd-lambdas", nargs="+", type=float,
                        default=[0.0, 0.1, 0.5],
                        help="MMD regularisation weights to try")
    args = parser.parse_args()
    main(args.csv, args.output, args.dev_batches, args.final_batches, args.mmd_lambdas)
