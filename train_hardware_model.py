"""Train a model for the actual ESP32-S3 hardware signals.

Expected CSV columns:
TVOC,eCO2,mems_acetone_ppm,temperature,humidity,label

label examples: clean, acetone, warning
Run:
    python train_hardware_model.py --csv hardware_data.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["TVOC", "eCO2", "mems_acetone_ppm", "temperature", "humidity"]

def load_data(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    required = FEATURES + ["label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少字段: {missing}")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    return df

def main(path: Path):
    df = load_data(path)
    if len(df) < 20:
        raise ValueError("有效样本少于20条，请先采集更多硬件数据")
    x, y = df[FEATURES], df["label"].astype(str)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(xtr, ytr); pred = clf.predict(xte)
    print("=== 硬件气体状态分类 ===")
    print(classification_report(yte, pred, zero_division=0))

    # Estimate MEMS acetone ppm from independent signals for consistency checking.
    reg_features = ["TVOC", "eCO2", "temperature", "humidity"]
    rxtr, rxte = xtr[reg_features], xte[reg_features]
    rr = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rr.fit(rxtr, df.loc[xtr.index, "mems_acetone_ppm"])
    # The primary concentration is the calibrated MEMS reading; this auxiliary model
    # estimates acetone from the other signals and helps detect inconsistent readings.
    aux_pred = rr.predict(rxte)
    actual = df.loc[xte.index, "mems_acetone_ppm"]
    print("=== 辅助丙酮浓度一致性检查 ===")
    print(f"MAE: {mean_absolute_error(actual, aux_pred):.3f} ppm")
    print(f"R2: {r2_score(actual, aux_pred):.3f}")
    joblib.dump({"classifier": clf, "aux_regressor": rr, "features": FEATURES,
                 "reg_features": reg_features}, "hardware_gas_model.joblib")
    print("模型已保存: hardware_gas_model.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path(__file__).parent / "hardware_data.csv")
    main(ap.parse_args().csv)
