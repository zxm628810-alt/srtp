"""Ablation study for the multimodal fusion DNN.

Tests 4 variants on the rolling temporal protocol (batches 4-10):
  full        – sensor branch + time branch + dropout + joint classification+regression loss
  no_time     – sensor branch only, no time features (baseline DNN in PyTorch)
  no_reg      – classification loss only, no regression auxiliary loss
  no_dropout  – same architecture but all Dropout layers removed

Each variant is trained and evaluated under identical conditions (same seed,
same train/val split, same early-stopping patience).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---- correct mapping (verified against prepare_uci_dataset.py) ----
GASES = {1: "Acetone", 2: "Acetaldehyde", 3: "Ethanol", 4: "Ethylene", 5: "Ammonia", 6: "Toluene"}
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
N_SENSORS = 128
N_BATCHES = 10
N_CLASSES = 6

# ---------------------------------------------------------------------------
def load_data(data_dir: Path):
    xs, gas_ids, ppm_values, batch_ids = [], [], [], []
    for path in sorted(data_dir.glob("batch*.dat")):
        batch_id = int(path.stem.replace("batch", ""))
        for line in path.read_text(errors="ignore").splitlines():
            parts = line.split()
            if not parts or ";" not in parts[0]:
                continue
            try:
                gas_id, ppm = parts[0].split(";", 1)
                values = [float(item.split(":", 1)[1]) for item in parts[1:129]]
                if len(values) == N_SENSORS:
                    xs.append(values)
                    gas_ids.append(int(gas_id))
                    ppm_values.append(float(ppm))
                    batch_ids.append(batch_id)
            except (ValueError, IndexError):
                continue
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(gas_ids, dtype=np.int64) - 1,
        np.asarray(ppm_values, dtype=np.float32),
        np.asarray(batch_ids, dtype=np.int64),
    )

def build_time_features(batch_ids: np.ndarray) -> np.ndarray:
    norm = (batch_ids - 1).astype(np.float32) / (N_BATCHES - 1)
    one_hot = np.zeros((len(batch_ids), N_BATCHES), dtype=np.float32)
    one_hot[np.arange(len(batch_ids)), batch_ids - 1] = 1.0
    return np.concatenate([norm.reshape(-1, 1), one_hot], axis=1)

# ---------------------------------------------------------------------------
class FusionDNN(nn.Module):
    """Full model with sensor branch, time branch, fusion layers, dropout."""
    def __init__(self, n_time: int, use_dropout: bool = True):
        super().__init__()
        drop1, drop2 = (0.35, 0.25) if use_dropout else (0.0, 0.0)
        drop_f1, drop_f2 = (0.20, 0.0) if use_dropout else (0.0, 0.0)
        self.sensor_branch = nn.Sequential(
            nn.Linear(N_SENSORS, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(drop2),
        )
        self.time_branch = nn.Sequential(
            nn.Linear(n_time, 32), nn.ReLU(), nn.Dropout(0.10 if use_dropout else 0.0),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 128), nn.ReLU(), nn.Dropout(drop_f1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop_f2),
        )
        self.classifier = nn.Linear(64, N_CLASSES)
        self.regressor = nn.Linear(64, 1)

    def forward(self, sensors, time_features):
        s = self.sensor_branch(sensors)
        t = self.time_branch(time_features)
        fused = self.fusion(torch.cat([s, t], dim=1))
        return self.classifier(fused), self.regressor(fused)


class SensorOnlyDNN(nn.Module):
    """Sensor-only variant: no time branch, fusion directly from sensor features."""
    def __init__(self, use_dropout: bool = True):
        super().__init__()
        drop1, drop2 = (0.35, 0.25) if use_dropout else (0.0, 0.0)
        self.sensor_branch = nn.Sequential(
            nn.Linear(N_SENSORS, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(drop2),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.20 if use_dropout else 0.0),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.classifier = nn.Linear(64, N_CLASSES)
        self.regressor = nn.Linear(64, 1)

    def forward(self, sensors, _time_features=None):
        s = self.sensor_branch(sensors)
        h = self.head(s)
        return self.classifier(h), self.regressor(h)

# ---------------------------------------------------------------------------
def train_one_fold(x_train, time_train, gas_train, ppm_train, *,
                   device, variant, epochs=80, batch_size=256, lr=1e-3,
                   patience=12, seed=42):
    """Returns (state_dict, sensor_mean, sensor_std) for the best checkpoint."""

    torch.manual_seed(seed); np.random.seed(seed)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True); std[std == 0] = 1.0
    x_train_s = (x_train - mean) / std
    ppm_train_log = np.log1p(ppm_train)

    train_idx, val_idx = train_test_split(
        np.arange(len(x_train)), test_size=0.15, random_state=seed, stratify=gas_train,
    )

    def ds(idx):
        x_t = torch.from_numpy(x_train_s[idx])
        t_t = torch.from_numpy(time_train[idx])
        g_t = torch.from_numpy(gas_train[idx])
        p_t = torch.from_numpy(ppm_train_log[idx])
        return TensorDataset(x_t, t_t, g_t, p_t)

    train_loader = DataLoader(ds(train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds(val_idx), batch_size=batch_size, shuffle=False)

    use_dropout = (variant != "no_dropout")
    n_time = time_train.shape[1]

    if variant in ("no_time",):
        model = SensorOnlyDNN(use_dropout=use_dropout)
    else:
        model = FusionDNN(n_time=n_time, use_dropout=use_dropout)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    reg_weight = 0.0 if variant == "no_reg" else 0.5
    best_val_acc = 0.0; best_state = None; no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for sensors_b, t_b, g_b, p_b in train_loader:
            sensors_b, t_b, g_b, p_b = sensors_b.to(device), t_b.to(device), g_b.to(device), p_b.to(device)
            opt.zero_grad()
            if variant == "no_time":
                logits_g, logits_p = model(sensors_b)
            else:
                logits_g, logits_p = model(sensors_b, t_b)
            loss = ce(logits_g, g_b)
            if reg_weight > 0:
                loss = loss + reg_weight * mse(logits_p.squeeze(1), p_b)
            loss.backward(); opt.step()

        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for sensors_b, t_b, g_b, _ in val_loader:
                sensors_b, g_b = sensors_b.to(device), g_b.to(device)
                if variant == "no_time":
                    logits_g, _ = model(sensors_b)
                else:
                    logits_g, _ = model(sensors_b, t_b.to(device))
                val_correct += (logits_g.argmax(dim=1) == g_b).sum().item()
                val_total += len(sensors_b)
        val_acc = val_correct / val_total
        sched.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    return best_state, mean, std, n_time


@torch.no_grad()
def predict(model, variant, x, time_feat, mean, std, device, batch_size=1024):
    model.eval(); model.to(device)
    x = (x - mean) / std
    preds = []
    for start in range(0, len(x), batch_size):
        s = torch.from_numpy(x[start:start+batch_size]).to(device)
        if variant == "no_time":
            logits, _ = model(s)
        else:
            t = torch.from_numpy(time_feat[start:start+batch_size]).to(device)
            logits, _ = model(s, t)
        preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


# ---------------------------------------------------------------------------
VARIANTS = {
    "full":        "sensor + time + dropout + joint loss",
    "no_time":     "sensor only (no time branch, no time features)",
    "no_reg":      "classification loss only (no regression auxiliary)",
    "no_dropout":  "all Dropout layers removed",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "uci_gas"/"unzipped")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "drift_results_ablation")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    x, gas_id, ppm, batch_id = load_data(args.data_dir)
    time_features = build_time_features(batch_id)

    all_rows = []

    for variant, description in VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"Ablation variant: {variant}  ({description})")
        print(f"{'='*70}")

        for test_batch in range(4, 11):
            train_mask = batch_id < test_batch
            test_mask = batch_id == test_batch
            n_tr = train_mask.sum(); n_te = test_mask.sum()

            state, mean, std, n_t = train_one_fold(
                x[train_mask], time_features[train_mask],
                gas_id[train_mask], ppm[train_mask],
                device=device, variant=variant,
                epochs=args.epochs, patience=args.patience,
            )

            if variant == "no_time":
                model = SensorOnlyDNN(use_dropout=True)
            elif variant == "no_dropout":
                model = FusionDNN(n_time=n_t, use_dropout=False)
            else:
                model = FusionDNN(n_time=n_t, use_dropout=True)
            model.load_state_dict(state)

            pred = predict(model, variant, x[test_mask], time_features[test_mask], mean, std, device)

            row = {
                "variant": variant, "description": description,
                "test_batch": test_batch, "n_train": n_tr, "n_test": n_te,
                "accuracy": accuracy_score(gas_id[test_mask], pred),
                "macro_f1": f1_score(gas_id[test_mask], pred, labels=list(range(N_CLASSES)), average="macro", zero_division=0),
            }
            recalls = recall_score(gas_id[test_mask], pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)
            row.update({f"recall_{name}": float(v) for name, v in zip(GAS_NAMES, recalls)})

            cm = pd.DataFrame(
                confusion_matrix(gas_id[test_mask], pred, labels=list(range(N_CLASSES))),
                index=GAS_NAMES, columns=GAS_NAMES,
            )
            cm.to_csv(args.output / f"confusion_ablation_{variant}_batch{test_batch}.csv", encoding="utf-8-sig")

            all_rows.append(row)
            print(f"  Batch {test_batch}: Acc={row['accuracy']:.4f}  F1={row['macro_f1']:.4f}")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.output / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nResults saved: {args.output.resolve()}")

    # ---- print comparison table ----
    print("\n\n==========  ABLATION COMPARISON  ==========")
    pivot = summary.pivot_table(
        index="test_batch", columns="variant",
        values=["accuracy", "macro_f1"], aggfunc="first"
    )
    print(pivot.to_string())


if __name__ == "__main__":
    main()
