"""Domain adaptation via MMD for gas sensor drift under rolling validation.

Problem: sensor readings drift over time, so the distribution of features from
early batches (source) differs from late batches (target).  A standard DNN only
minimises classification error on the training (source) distribution; when the
test distribution shifts, performance drops.

Idea: add an MMD (Maximum Mean Discrepancy) regularisation term to the training
loss that encourages the encoder to produce features that are *batch-invariant*
— the distribution of encoded features from old batches should look like the
distribution from recent batches.  This is unsupervised domain adaptation: the
target domain (most recent training batch) is used without labels.

MMD² = E[k(z_s, z_s')] + E[k(z_t, z_t')] - 2·E[k(z_s, z_t)]

where z are encoded features (the 64-dim hidden layer before the classifier),
k is an RBF kernel, and s=source (oldest training batch), t=target (newest
training batch).  Lower MMD means the two distributions are more similar.

Training loss = CrossEntropy + λ * MMD²

The experiment:
  1. Train a baseline PyTorch MLP (same architecture as sklearn version)
  2. Train the same MLP with MMD regularisation
  3. Compare under strict rolling validation (batches 4-10)
  4. Select λ on dev set (batches 4-8), confirm on batches 9-10
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent
FEATURES = [f"feature_{i}" for i in range(1, 129)]
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
LOW_PPM = 50.0
DEFAULT_SEED = 42

# ---- MMD kernel (RBF / Gaussian) ----
def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Pairwise RBF kernel matrix between two sets of vectors."""
    x_norm = (x ** 2).sum(1).unsqueeze(1)       # (n, 1)
    y_norm = (y ** 2).sum(1).unsqueeze(0)       # (1, m)
    dist = x_norm + y_norm - 2.0 * x @ y.T      # (n, m)
    return torch.exp(-dist / (2.0 * sigma ** 2))


def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Unbiased estimate of MMD² between two sets of encoded features."""
    k_xx = gaussian_kernel(x, x, sigma)
    k_yy = gaussian_kernel(y, y, sigma)
    k_xy = gaussian_kernel(x, y, sigma)

    n, m = x.size(0), y.size(0)
    # Remove diagonal for unbiased estimate
    k_xx = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1)) if n > 1 else k_xx.mean()
    k_yy = (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1)) if m > 1 else k_yy.mean()
    k_xy = k_xy.mean()

    return k_xx + k_yy - 2 * k_xy


# ---- Multi-sigma MMD (blend of bandwidths for robustness) ----
def multi_sigma_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Average MMD over several RBF bandwidths for better gradient signal."""
    # Compute median pairwise distance for automatic sigma selection
    with torch.no_grad():
        combined = torch.cat([x, y], dim=0)
        dists = torch.cdist(combined, combined)
        median_dist = dists[dists > 0].median()

    sigmas = [median_dist * s for s in [0.5, 1.0, 2.0]]
    return sum(mmd_loss(x, y, sigma=s) for s in sigmas) / len(sigmas)


# ---- Model ----
class EncoderMLP(nn.Module):
    """3-layer MLP matching the sklearn version: 128 → 256 → 128 → 64 → 6."""
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor, return_encoded: bool = False):
        z = self.encoder(x)
        logits = self.classifier(z)
        if return_encoded:
            return logits, z
        return logits


def seed_everything(seed: int) -> None:
    """Set all stochastic sources for a reproducible experimental run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(x: np.ndarray, y: np.ndarray, batch_ids: np.ndarray,
                mmd_lambda: float, lr: float = 1e-3, epochs: int = 250,
                patience: int = 25, seed: int = DEFAULT_SEED) -> EncoderMLP:
    """Train with optional MMD regularisation (mmd_lambda=0 = baseline)."""

    seed_everything(seed)
    train_idx, valid_idx = train_test_split(
        np.arange(len(x)), test_size=.15, random_state=seed, stratify=y,
    )

    n_features, n_classes = x.shape[1], len(np.unique(y))
    model = EncoderMLP(n_features, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    x_train_t = torch.tensor(x[train_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.long)
    b_train_t = torch.tensor(batch_ids[train_idx], dtype=torch.long)

    x_valid_t = torch.tensor(x[valid_idx], dtype=torch.float32)
    y_valid_t = torch.tensor(y[valid_idx], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t, b_train_t),
        batch_size=256, shuffle=True,
    )

    best_state, best_loss, no_improve = None, float("inf"), 0

    for _ in range(epochs):
        model.train()
        for xb, yb, bb in train_loader:
            opt.zero_grad()

            if mmd_lambda > 0:
                logits, z = model(xb, return_encoded=True)
                cls_loss = ce(logits, yb)

                # Source = oldest batch in this mini-batch, target = newest
                # Split by median batch ID within the mini-batch
                median_batch = bb.median()
                src_mask = bb <= median_batch
                tgt_mask = bb > median_batch

                if src_mask.sum() >= 2 and tgt_mask.sum() >= 2:
                    mmd = multi_sigma_mmd(z[src_mask], z[tgt_mask])
                    loss = cls_loss + mmd_lambda * mmd
                else:
                    loss = cls_loss
            else:
                logits = model(xb)
                loss = ce(logits, yb)

            loss.backward()
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(x_valid_t)
            val_loss = ce(logits, y_valid_t).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_state = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


def evaluate(train: pd.DataFrame, test: pd.DataFrame, mmd_lambda: float,
             output: Path, seed: int = DEFAULT_SEED) -> dict:
    """Train on batches < k, test on batch k."""

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES].values)
    x_test = scaler.transform(test[FEATURES].values)

    class_to_id = {g: i for i, g in enumerate(GAS_NAMES)}
    y_train = train.gas_name.map(class_to_id).values
    batch_ids_train = train.batch_id.values

    model = train_model(x_train, y_train, batch_ids_train, mmd_lambda, seed=seed)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32))
        pred_ids = logits.argmax(dim=1).numpy()

    pred_gas = np.array(GAS_NAMES)[pred_ids]
    batch = int(test.batch_id.iloc[0])
    low = test.concentration_ppm <= LOW_PPM

    row = {
        "model": "dnn_mmd" if mmd_lambda > 0 else "dnn_baseline",
        "mmd_lambda": mmd_lambda,
        "seed": seed,
        "test_batch": batch,
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": accuracy_score(test.gas_name, pred_gas),
        "macro_f1": f1_score(test.gas_name, pred_gas, labels=GAS_NAMES, average="macro", zero_division=0),
    }
    recalls = recall_score(test.gas_name, pred_gas, labels=GAS_NAMES, average=None, zero_division=0)
    row.update({f"recall_{g}": float(v) for g, v in zip(GAS_NAMES, recalls)})
    row["low_ppm_n"] = int(low.sum())
    row["low_ppm_accuracy"] = accuracy_score(
        test.loc[low, "gas_name"], pred_gas[low]
    ) if low.any() else np.nan

    return row


def main(csv_path: Path, output: Path, dev_batches: list[int],
         final_batches: list[int], lambdas: list[float], seed: int = DEFAULT_SEED) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # ===== Step 1: Dev phase — select λ on batches 4-8 =====
    print("=" * 70)
    print("DEV PHASE: Select λ on batches 4-8")
    print("=" * 70)

    dev_rows = []
    for mmd_lambda in lambdas:
        label = f"λ={mmd_lambda}" if mmd_lambda > 0 else "Baseline (no MMD)"
        print(f"\n--- {label} ---")
        for batch in dev_batches:
            train = df[df["batch_id"] < batch]
            test = df[df["batch_id"] == batch]
            row = evaluate(train, test, mmd_lambda, output, seed=seed)
            dev_rows.append(row)
            print(f"  Batch {batch}: Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}")

    dev_results = pd.DataFrame(dev_rows)
    dev_summary = dev_results.groupby("mmd_lambda").agg(
        avg_accuracy=("accuracy", "mean"),
        avg_f1=("macro_f1", "mean"),
    ).reset_index()
    dev_summary["rank_acc"] = dev_summary["avg_accuracy"].rank(ascending=False)
    dev_summary["rank_f1"] = dev_summary["avg_f1"].rank(ascending=False)
    dev_summary["mean_rank"] = (dev_summary["rank_acc"] + dev_summary["rank_f1"]) / 2

    print(f"\n{'=' * 70}")
    print("DEV SUMMARY")
    print("=" * 70)
    print(dev_summary.to_string(index=False))

    best_lambda = dev_summary.loc[dev_summary["mean_rank"].idxmin(), "mmd_lambda"]
    print(f"\nSelected λ = {best_lambda}")

    # ===== Step 2: Final confirm on batches 9-10 =====
    print(f"\n{'=' * 70}")
    print(f"FINAL PHASE: Confirm λ={best_lambda} on batches 9-10")
    print("=" * 70)

    final_rows = []
    # Baseline
    print("\n--- Baseline (no MMD) ---")
    for batch in final_batches:
        train = df[df["batch_id"] < batch]
        test = df[df["batch_id"] == batch]
        row = evaluate(train, test, mmd_lambda=0.0, output=output, seed=seed)
        final_rows.append(row)
        print(f"  Batch {batch}: Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}")

    # Best MMD
    print(f"\n--- MMD λ={best_lambda} ---")
    for batch in final_batches:
        train = df[df["batch_id"] < batch]
        test = df[df["batch_id"] == batch]
        row = evaluate(train, test, mmd_lambda=best_lambda, output=output, seed=seed)
        final_rows.append(row)
        print(f"  Batch {batch}: Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}")

    # ===== Save =====
    all_results = pd.concat([dev_results, pd.DataFrame(final_rows)], ignore_index=True)
    all_results.to_csv(output / "domain_adaptation_results.csv", index=False, encoding="utf-8-sig")
    print(f"\nResults saved: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "domain_adaptation_results")
    parser.add_argument("--dev-batches", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--final-batches", nargs="+", type=int, default=[9, 10])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.1, 0.5, 1.0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for one independent run.")
    args = parser.parse_args()
    main(args.csv, args.output, args.dev_batches, args.final_batches, args.lambdas, args.seed)
