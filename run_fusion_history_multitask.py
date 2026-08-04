"""Fusion multi-task DNN with train-history calibration, no test batch ID.

The sensor branch uses raw 128 UCI features.  The history branch uses 128
relative-to-median features calculated from the latest three earlier batches.
Both branches are trained jointly for six-class gas classification and ppm
regression under the strict rolling protocol.
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

from run_drift_experiments import FEATURES, GAS_NAMES, select_x


SEED = 42
LOW_PPM = 50.0


class FusionHistoryNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sensor_branch = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(.20),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.15),
        )
        self.history_branch = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(.10), nn.Linear(128, 64), nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Linear(64, len(GAS_NAMES)); self.regressor = nn.Linear(64, 1)

    def forward(self, sensor: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.fusion(torch.cat([self.sensor_branch(sensor), self.history_branch(history)], dim=1))
        return self.classifier(features), self.regressor(features).squeeze(1)


def seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def inputs(frame: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_x = select_x(frame, "dnn_sensor_history_baseline", history=history, history_window=3)
    return all_x[FEATURES], all_x[[f"history_relative_{name}" for name in FEATURES]]


def train(sensor: np.ndarray, history: np.ndarray, classes: np.ndarray, ppm: np.ndarray, reg_weight: float) -> tuple[FusionHistoryNet, StandardScaler]:
    ppm_scaler = StandardScaler(); y_reg = ppm_scaler.fit_transform(ppm.reshape(-1, 1)).ravel()
    train_idx, valid_idx = train_test_split(np.arange(len(sensor)), test_size=.15, random_state=SEED, stratify=classes)
    model = FusionHistoryNet(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce, huber = nn.CrossEntropyLoss(), nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(torch.tensor(sensor[train_idx], dtype=torch.float32), torch.tensor(history[train_idx], dtype=torch.float32),
                                      torch.tensor(classes[train_idx], dtype=torch.long), torch.tensor(y_reg[train_idx], dtype=torch.float32)), batch_size=256, shuffle=True)
    sv = torch.tensor(sensor[valid_idx], dtype=torch.float32); hv = torch.tensor(history[valid_idx], dtype=torch.float32)
    cv = torch.tensor(classes[valid_idx], dtype=torch.long); rv = torch.tensor(y_reg[valid_idx], dtype=torch.float32)
    best, best_loss, patience = None, float("inf"), 0
    for _ in range(250):
        model.train()
        for sb, hb, cb, rb in loader:
            logits, reg = model(sb, hb); loss = ce(logits, cb) + reg_weight * huber(reg, rb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, reg = model(sv, hv); valid_loss = (ce(logits, cv) + reg_weight * huber(reg, rv)).item()
        if valid_loss < best_loss - 1e-5:
            best_loss, patience = valid_loss, 0; best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 25: break
    model.load_state_dict(best); return model, ppm_scaler


def evaluate(train_frame: pd.DataFrame, test_frame: pd.DataFrame, output: Path, reg_weight: float) -> dict:
    seed(); s_train, h_train = inputs(train_frame, train_frame); s_test, h_test = inputs(test_frame, train_frame)
    s_scale, h_scale = StandardScaler(), StandardScaler()
    s_train, h_train = s_scale.fit_transform(s_train), h_scale.fit_transform(h_train)
    s_test, h_test = s_scale.transform(s_test), h_scale.transform(h_test)
    class_ids = train_frame.gas_name.map({gas: i for i, gas in enumerate(GAS_NAMES)}).to_numpy()
    model, ppm_scale = train(s_train, h_train, class_ids, train_frame.concentration_ppm.to_numpy(), reg_weight)
    model.eval()
    with torch.no_grad():
        logits, y_reg = model(torch.tensor(s_test, dtype=torch.float32), torch.tensor(h_test, dtype=torch.float32))
        gas = np.array(GAS_NAMES)[logits.argmax(1).numpy()]
        ppm = np.maximum(ppm_scale.inverse_transform(y_reg.numpy().reshape(-1, 1)).ravel(), 0)
    batch = int(test_frame.batch_id.iloc[0]); low = test_frame.concentration_ppm <= LOW_PPM
    row = {"model": "fusion_history_multitask", "regression_weight": reg_weight, "test_batch": batch, "n_train": len(train_frame), "n_test": len(test_frame),
           "accuracy": accuracy_score(test_frame.gas_name, gas), "macro_f1": f1_score(test_frame.gas_name, gas, labels=GAS_NAMES, average="macro", zero_division=0),
           "mae": mean_absolute_error(test_frame.concentration_ppm, ppm), "r2": r2_score(test_frame.concentration_ppm, ppm), "low_ppm_n": int(low.sum()),
           "low_ppm_mae": mean_absolute_error(test_frame.loc[low, "concentration_ppm"], ppm[low]) if low.any() else np.nan}
    for name in GAS_NAMES:
        mask = test_frame.gas_name == name; row[f"mae_{name}"] = mean_absolute_error(test_frame.loc[mask, "concentration_ppm"], ppm[mask]) if mask.any() else np.nan
    details = test_frame[["batch_id", "gas_name", "concentration_ppm"]].copy(); details["predicted_gas"] = gas; details["predicted_ppm"] = ppm
    details["absolute_error_ppm"] = np.abs(details.concentration_ppm - ppm)
    details.to_csv(output / f"predictions_fusion_history_multitask_batch{batch}.csv", index=False, encoding="utf-8-sig")
    return row


def main(csv: Path, output: Path, batches: list[int], reg_weight: float) -> None:
    output.mkdir(parents=True, exist_ok=True); data = pd.read_csv(csv, encoding="utf-8-sig")
    result = pd.DataFrame([evaluate(data[data.batch_id < batch], data[data.batch_id == batch], output, reg_weight) for batch in batches])
    result.to_csv(output / "cross_batch_fusion_history_multitask_metrics.csv", index=False, encoding="utf-8-sig")
    print(result[["test_batch", "accuracy", "macro_f1", "mae", "r2", "low_ppm_mae"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "all_batches.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "fusion_history_multitask_results")
    parser.add_argument("--test-batches", nargs="+", type=int, choices=range(4, 11), default=list(range(4, 11)))
    parser.add_argument("--regression-weight", type=float, default=.60)
    args = parser.parse_args(); main(args.csv, args.output, args.test_batches, args.regression_weight)
