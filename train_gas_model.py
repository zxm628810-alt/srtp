"""Train an MQ-sensor gas classifier and expose a hardware-friendly predictor.

Usage:
    python train_gas_model.py --csv Gas_Sensors_Measurements.csv
    python train_gas_model.py --predict 120 35 80 20 10 15 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

SENSORS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]
MODEL_PATH = Path("gas_classifier.joblib")


def read_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in SENSORS + ["Gas"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少字段: {missing}; 实际字段为: {list(df.columns)}")
    x = df[SENSORS].apply(pd.to_numeric, errors="coerce")
    y = df["Gas"].astype(str).str.strip()
    valid = x.notna().any(axis=1) & y.ne("") & y.ne("nan")
    return x.loc[valid], y.loc[valid]


def train(csv_path: Path) -> None:
    x, y_text = read_dataset(csv_path)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(classification_report(y_test, pred, target_names=encoder.classes_, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    artifact = {"model": model, "label_encoder": encoder, "sensors": SENSORS}
    joblib.dump(artifact, MODEL_PATH)
    print(f"模型已保存: {MODEL_PATH.resolve()}")
    print("类别:", list(encoder.classes_))


def predict(values: list[float]) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 {MODEL_PATH}，请先训练模型")
    if len(values) != len(SENSORS):
        raise ValueError(f"需要7个传感器值，顺序为: {SENSORS}")
    artifact = joblib.load(MODEL_PATH)
    model, encoder = artifact["model"], artifact["label_encoder"]
    row = np.asarray(values, dtype=float).reshape(1, -1)
    cls = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    result = {
        "gas": str(encoder.inverse_transform([cls])[0]),
        "confidence": round(float(probabilities[cls]), 4),
        "sensor_values": dict(zip(SENSORS, values)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, help="Gas_Sensors_Measurements.csv")
    parser.add_argument("--predict", nargs=7, type=float, metavar="SENSOR")
    args = parser.parse_args()
    if args.csv:
        train(args.csv)
    elif args.predict:
        predict(args.predict)
    else:
        parser.error("请提供 --csv 或 --predict")
