"""Strict rolling multi-task DNN for gas class and concentration together.

The shared encoder receives raw 128 sensor features plus 128 features relative
to the latest three historical batches.  It has two heads: six-way gas
classification and ppm regression.  For Batch k, every scaler, validation split
and calibration feature uses Batch 1..k-1 only.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_drift_experiments import GAS_NAMES, select_x


SEED = 42
LOW_PPM = 50.0


class MultiTaskNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(.15),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.10),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.classifier = nn.Linear(64, n_classes)
        self.regressor = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.classifier(encoded), self.regressor(encoded).squeeze(1)


def seed_everything() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def feature_matrix(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    return select_x(frame, "dnn_sensor_history_baseline", history=history, history_window=3)


def train_model(x: np.ndarray, y_class: np.ndarray, y_ppm: np.ndarray, regression_weight: float) -> tuple[MultiTaskNet, StandardScaler]:
    target_scaler = StandardScaler(); y_scaled = target_scaler.fit_transform(y_ppm.reshape(-1, 1)).ravel()
    idx = np.arange(len(x))
    train_idx, valid_idx = train_test_split(idx, test_size=.15, random_state=SEED, stratify=y_class)
    model = MultiTaskNet(x.shape[1], len(GAS_NAMES)); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce, huber = nn.CrossEntropyLoss(), nn.SmoothL1Loss()
    train_loader = DataLoader(TensorDataset(torch.tensor(x[train_idx], dtype=torch.float32),
                                            torch.tensor(y_class[train_idx], dtype=torch.long),
                                            torch.tensor(y_scaled[train_idx], dtype=torch.float32)), batch_size=256, shuffle=True)
    x_valid = torch.tensor(x[valid_idx], dtype=torch.float32); yc_valid = torch.tensor(y_class[valid_idx], dtype=torch.long); yr_valid = torch.tensor(y_scaled[valid_idx], dtype=torch.float32)
    best_state, best_loss, patience = None, float("inf"), 0
    for _ in range(250):
        model.train()
        for xb, ycb, yrb in train_loader:
            logits, reg = model(xb); loss = ce(logits, ycb) + regression_weight * huber(reg, yrb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, reg = model(x_valid); valid_loss = (ce(logits, yc_valid) + regression_weight * huber(reg, yr_valid)).item()
        if valid_loss < best_loss - 1e-5:
            best_loss, patience = valid_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 25: break
    model.load_state_dict(best_state); return model, target_scaler


def evaluate(train: pd.DataFrame, test: pd.DataFrame, output: Path, regression_weight: float) -> dict:
    seed_everything()
    raw_train, raw_test = feature_matrix(train, train), feature_matrix(test, train)
    feature_scaler = StandardScaler(); x_train = feature_scaler.fit_transform(raw_train); x_test = feature_scaler.transform(raw_test)
    class_to_id = {gas: i for i, gas in enumerate(GAS_NAMES)}
    y_class = train.gas_name.map(class_to_id).to_numpy()
    model, target_scaler = train_model(x_train, y_class, train.concentration_ppm.to_numpy(), regression_weight)
    model.eval()
    with torch.no_grad():
        logits, scaled_ppm = model(torch.tensor(x_test, dtype=torch.float32))
        predicted_class = logits.argmax(dim=1).numpy()
        predicted_ppm = np.maximum(target_scaler.inverse_transform(scaled_ppm.numpy().reshape(-1, 1)).ravel(), 0)
    predicted_gas = np.array(GAS_NAMES)[predicted_class]; batch = int(test.batch_id.iloc[0]); low = test.concentration_ppm <= LOW_PPM
    row = {"model": "multitask_dnn_history_w3", "regression_weight": regression_weight, "test_batch": batch, "n_train": len(train), "n_test": len(test),
           "accuracy": accuracy_score(test.gas_name, predicted_gas), "macro_f1": f1_score(test.gas_name, predicted_gas, labels=GAS_NAMES, average="macro", zero_division=0),
           "mae": mean_absolute_error(test.concentration_ppm, predicted_ppm), "r2": r2_score(test.concentration_ppm, predicted_ppm),
           "low_ppm_n": int(low.sum()), "low_ppm_mae": mean_absolute_error(test.loc[low, "concentration_ppm"], predicted_ppm[low]) if low.any() else np.nan}
    for gas in GAS_NAMES:
        mask = test.gas_name == gas
        row[f"mae_{gas}"] = mean_absolute_error(test.loc[mask, "concentration_ppm"], predicted_ppm[mask]) if mask.any() else np.nan
    details = test[["batch_id", "gas_name", "concentration_ppm"]].copy()
    details["predicted_gas"] = predicted_gas; details["predicted_ppm"] = predicted_ppm; details["absolute_error_ppm"] = np.abs(details.concentration_ppm - predicted_ppm)
    details.to_csv(output / f"predictions_multitask_dnn_batch{batch}.csv", index=False, encoding="utf-8-sig")
    return row


def main(csv: Path, output: Path, batches: list[int], regression_weight: float) -> None:
    output.mkdir(parents=True, exist_ok=True); data = pd.read_csv(csv, encoding="utf-8-sig")
    rows = [evaluate(data[data.batch_id < batch], data[data.batch_id == batch], output, regression_weight) for batch in batches]
    table = pd.DataFrame(rows); table.to_csv(output / "cross_batch_multitask_metrics.csv", index=False, encoding="utf-8-sig")
    print(table[["test_batch", "accuracy", "macro_f1", "mae", "r2", "low_ppm_mae"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "multitask_dnn_results")
    parser.add_argument("--test-batches", nargs="+", type=int, choices=range(4, 11), default=list(range(4, 11)))
    parser.add_argument("--regression-weight", type=float, default=.35,
                        help="weight of standardized concentration Smooth L1 loss")
    args = parser.parse_args(); main(args.csv, args.output, args.test_batches, args.regression_weight)
