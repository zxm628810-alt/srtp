"""Single-branch adversarial domain adaptation (DANN) under rolling drift tests.

The encoder is trained for gas classification while a gradient-reversal domain
head tries to predict the *historical training batch*.  The encoder therefore
learns representations that retain gas information but reduce batch identity.
No test-batch ID is used at inference.
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
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent
FEATURES = [f"feature_{i}" for i in range(1, 129)]
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
LOW_PPM = 50.0
SEED = 42


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * grad_output, None


def gradient_reverse(x: torch.Tensor, coefficient: float) -> torch.Tensor:
    return GradientReverse.apply(x, coefficient)


class DANN(nn.Module):
    def __init__(self, n_features: int, n_classes: int, n_domains: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.gas_head = nn.Linear(64, n_classes)
        self.domain_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_domains))

    def forward(self, x: torch.Tensor, grl_coefficient: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        gas_logits = self.gas_head(z)
        domain_logits = self.domain_head(gradient_reverse(z, grl_coefficient))
        return gas_logits, domain_logits


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_model(x: np.ndarray, y: np.ndarray, batches: np.ndarray, domain_lambda: float,
                seed: int = SEED) -> DANN:
    seed_everything(seed)
    idx = np.arange(len(x))
    train_idx, valid_idx = train_test_split(idx, test_size=.15, random_state=seed, stratify=y)
    domain_map = {batch: i for i, batch in enumerate(sorted(np.unique(batches)))}
    domains = np.array([domain_map[b] for b in batches], dtype=np.int64)
    model = DANN(x.shape[1], len(GAS_NAMES), len(domain_map))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gas_loss = nn.CrossEntropyLoss()
    domain_loss = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.long),
            torch.tensor(domains[train_idx], dtype=torch.long),
        ), batch_size=256, shuffle=True,
    )
    xv = torch.tensor(x[valid_idx], dtype=torch.float32)
    yv = torch.tensor(y[valid_idx], dtype=torch.long)
    best_state, best_val, patience = None, float("inf"), 0
    for _ in range(250):
        model.train()
        for xb, yb, db in loader:
            opt.zero_grad()
            gas_logits, domain_logits = model(xb, grl_coefficient=domain_lambda)
            # The gradient-reversal coefficient already scales the adversarial
            # gradient received by the encoder.  Keep the domain-head loss at
            # unit scale so lambda is not applied twice.
            loss = gas_loss(gas_logits, yb) + domain_loss(domain_logits, db)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            valid_logits, _ = model(xv)
            val = gas_loss(valid_logits, yv).item()
        if val < best_val - 1e-5:
            best_val, best_state, patience = val, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 25:
                break
    model.load_state_dict(best_state)
    return model


def evaluate(train: pd.DataFrame, test: pd.DataFrame, domain_lambda: float, seed: int = SEED) -> dict:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES].values)
    x_test = scaler.transform(test[FEATURES].values)
    class_map = {g: i for i, g in enumerate(GAS_NAMES)}
    y_train = train.gas_name.map(class_map).values
    model = train_model(x_train, y_train, train.batch_id.values, domain_lambda, seed=seed)
    with torch.no_grad():
        logits, _ = model(torch.tensor(x_test, dtype=torch.float32))
        pred = np.array(GAS_NAMES)[logits.argmax(1).numpy()]
    low = test.concentration_ppm.values <= LOW_PPM
    recalls = recall_score(test.gas_name, pred, labels=GAS_NAMES, average=None, zero_division=0)
    row = {
        "model": "dann" if domain_lambda else "dnn_baseline",
        "domain_lambda": domain_lambda, "test_batch": int(test.batch_id.iloc[0]), "seed": seed,
        "accuracy": accuracy_score(test.gas_name, pred),
        "macro_f1": f1_score(test.gas_name, pred, labels=GAS_NAMES, average="macro", zero_division=0),
        "low_ppm_accuracy": accuracy_score(test.gas_name[low], pred[low]) if low.any() else np.nan,
    }
    row.update({f"recall_{gas}": float(value) for gas, value in zip(GAS_NAMES, recalls)})
    return row


def main(csv_path: Path, output: Path, lambdas: list[float], seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    dev_rows = []
    for lam in lambdas:
        print(f"\nDEV lambda={lam}")
        for batch in range(4, 9):
            row = evaluate(df[df.batch_id < batch], df[df.batch_id == batch], lam, seed)
            dev_rows.append(row)
            print(f"  Batch {batch}: Acc={row['accuracy']:.4f}, F1={row['macro_f1']:.4f}")
    dev = pd.DataFrame(dev_rows)
    summary = dev.groupby("domain_lambda").agg(avg_accuracy=("accuracy", "mean"), avg_f1=("macro_f1", "mean")).reset_index()
    summary["mean_rank"] = (summary.avg_accuracy.rank(ascending=False) + summary.avg_f1.rank(ascending=False)) / 2
    summary.to_csv(output / "dann_dev_selection.csv", index=False, encoding="utf-8-sig")
    best_lambda = float(summary.loc[summary.mean_rank.idxmin(), "domain_lambda"])
    print("Selected lambda:", best_lambda)

    final_rows = []
    for lam in [0.0, best_lambda]:
        for batch in [9, 10]:
            row = evaluate(df[df.batch_id < batch], df[df.batch_id == batch], lam, seed)
            row["phase"] = "final"
            final_rows.append(row)
            print(f"FINAL lambda={lam}, Batch {batch}: Acc={row['accuracy']:.4f}, F1={row['macro_f1']:.4f}")
    pd.concat([dev.assign(phase="dev"), pd.DataFrame(final_rows)], ignore_index=True).to_csv(
        output / "dann_results.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "dann_results")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.csv, args.output, args.lambdas, args.seed)
