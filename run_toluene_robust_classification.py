"""Toluene-focused strict rolling classification with Fusion-history features.

Variants:
  ce       ordinary cross entropy
  weighted class-weighted CE (Toluene loss weight 2)
  focal    class-weighted focal loss (gamma=2, Toluene weight 2)

The model never receives the current batch ID.  Its two branches process raw
sensor features and history-window-3 calibration features respectively.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_drift_experiments import FEATURES, GAS_NAMES, select_x


SEED = 42
TOLUENE_INDEX = GAS_NAMES.index("Toluene")


class FusionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sensor = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(.20),
                                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.15))
        self.history = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(.10), nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(GAS_NAMES)))

    def forward(self, sensor: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.sensor(sensor), self.history(history)], dim=1))


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits, targets, weight=weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt).pow(gamma) * ce).mean()


def make_inputs(frame: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = select_x(frame, "dnn_sensor_history_baseline", history=history, history_window=3)
    return merged[FEATURES], merged[[f"history_relative_{feature}" for feature in FEATURES]]


def train(sensor: np.ndarray, history: np.ndarray, labels: np.ndarray, variant: str) -> FusionClassifier:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    train_idx, valid_idx = train_test_split(np.arange(len(labels)), test_size=.15, random_state=SEED, stratify=labels)
    weights = torch.ones(len(GAS_NAMES));
    if variant in {"weighted", "focal"}: weights[TOLUENE_INDEX] = 2.0
    model = FusionClassifier(); optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.tensor(sensor[train_idx], dtype=torch.float32), torch.tensor(history[train_idx], dtype=torch.float32),
                                      torch.tensor(labels[train_idx], dtype=torch.long)), batch_size=256, shuffle=True)
    sv = torch.tensor(sensor[valid_idx], dtype=torch.float32); hv = torch.tensor(history[valid_idx], dtype=torch.float32); yv = torch.tensor(labels[valid_idx], dtype=torch.long)
    best, best_loss, patience = None, float("inf"), 0
    for _ in range(250):
        model.train()
        for sb, hb, yb in loader:
            logits = model(sb, hb)
            loss = focal_loss(logits, yb, weights) if variant == "focal" else nn.functional.cross_entropy(logits, yb, weight=weights)
            optim.zero_grad(); loss.backward(); optim.step()
        model.eval()
        with torch.no_grad():
            logits = model(sv, hv)
            val_loss = focal_loss(logits, yv, weights) if variant == "focal" else nn.functional.cross_entropy(logits, yv, weight=weights)
        if val_loss.item() < best_loss - 1e-5:
            best_loss, patience = val_loss.item(), 0; best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 25: break
    model.load_state_dict(best); return model


def evaluate(train_frame: pd.DataFrame, test_frame: pd.DataFrame, variant: str, output: Path) -> dict:
    sensor_train, history_train = make_inputs(train_frame, train_frame); sensor_test, history_test = make_inputs(test_frame, train_frame)
    ss, hs = StandardScaler(), StandardScaler()
    sensor_train, history_train = ss.fit_transform(sensor_train), hs.fit_transform(history_train)
    sensor_test, history_test = ss.transform(sensor_test), hs.transform(history_test)
    ids = train_frame.gas_name.map({gas: index for index, gas in enumerate(GAS_NAMES)}).to_numpy()
    model = train(sensor_train, history_train, ids, variant); model.eval()
    with torch.no_grad(): pred_ids = model(torch.tensor(sensor_test, dtype=torch.float32), torch.tensor(history_test, dtype=torch.float32)).argmax(1).numpy()
    predicted = np.array(GAS_NAMES)[pred_ids]; batch = int(test_frame.batch_id.iloc[0])
    recalls = recall_score(test_frame.gas_name, predicted, labels=GAS_NAMES, average=None, zero_division=0)
    row = {"variant": variant, "test_batch": batch, "accuracy": accuracy_score(test_frame.gas_name, predicted),
           "macro_f1": f1_score(test_frame.gas_name, predicted, labels=GAS_NAMES, average="macro", zero_division=0)}
    row.update({f"recall_{gas}": value for gas, value in zip(GAS_NAMES, recalls)})
    cm = pd.DataFrame(confusion_matrix(test_frame.gas_name, predicted, labels=GAS_NAMES), index=GAS_NAMES, columns=GAS_NAMES)
    cm.to_csv(output / f"confusion_toluene_{variant}_batch{batch}.csv", encoding="utf-8-sig")
    return row


def main(csv: Path, output: Path, variants: list[str], batches: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True); data = pd.read_csv(csv, encoding="utf-8-sig"); rows = []
    for variant in variants:
        for batch in batches:
            row = evaluate(data[data.batch_id < batch], data[data.batch_id == batch], variant, output); rows.append(row)
            print(f"{variant:8} Batch {batch}: Acc={row['accuracy']:.4f}, Toluene recall={row['recall_Toluene']:.4f}")
    pd.DataFrame(rows).to_csv(output / "toluene_robust_metrics.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "toluene_robust_results")
    parser.add_argument("--variants", nargs="+", choices=["ce", "weighted", "focal"], default=["ce", "weighted", "focal"])
    parser.add_argument("--test-batches", nargs="+", type=int, choices=range(4, 11), default=list(range(4, 11)))
    args = parser.parse_args(); main(args.csv, args.output, args.variants, args.test_batches)
