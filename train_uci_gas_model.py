"""Train a gas identity/concentration model from UCI batch*.dat files.

Put batch1.dat ... batch10.dat in the same directory, then run:
    python train_uci_gas_model.py --data-dir C:\\srtp\\uci_gas
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# The raw class IDs are verified against the official per-batch sample counts.
GASES = {1: "Acetone", 2: "Acetaldehyde", 3: "Ethanol", 4: "Ethylene", 5: "Ammonia", 6: "Toluene"}

def load(data_dir: Path):
    xs, labels, concentrations = [], [], []
    for path in sorted(data_dir.glob("batch*.dat")):
        for line in path.read_text(errors="ignore").splitlines():
            parts = line.split()
            if not parts or ";" not in parts[0]:
                continue
            try:
                gas_id, ppm = parts[0].split(";", 1)
                values = [float(item.split(":", 1)[1]) for item in parts[1:129]]
                if len(values) == 128:
                    labels.append(int(gas_id)); concentrations.append(float(ppm)); xs.append(values)
            except (ValueError, IndexError):
                continue
    if not xs:
        raise ValueError("没有找到有效 batch*.dat 文件")
    return np.asarray(xs), np.asarray(labels), np.asarray(concentrations)

def main(data_dir: Path):
    x, y, ppm = load(data_dir)
    xtr, xte, ytr, yte, ptr, pte = train_test_split(x, y, ppm, test_size=.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(xtr, ytr)
    pred = clf.predict(xte)
    names = [GASES[i] for i in sorted(np.unique(y))]
    print(classification_report(yte, pred, labels=sorted(np.unique(y)), target_names=names, zero_division=0))
    reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    reg.fit(xtr, ptr)
    pp = reg.predict(xte)
    print(f"浓度 MAE: {mean_absolute_error(pte, pp):.3f} ppm")
    print(f"浓度 R2: {r2_score(pte, pp):.3f}")
    joblib.dump({"classifier": clf, "regressor": reg, "gases": GASES}, "uci_gas_model.joblib")
    print("模型已保存: uci_gas_model.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "uci_gas" / "unzipped")
    main(ap.parse_args().data_dir)
