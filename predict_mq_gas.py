"""Human-readable single-sample test for the MQ gas model."""
from pathlib import Path
import argparse
import joblib
import numpy as np
from train_gas_model import read_dataset, SENSORS

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", type=Path, required=True)
ap.add_argument("--index", type=int, default=0, help="样本编号，从0开始")
args = ap.parse_args()

artifact = joblib.load("gas_classifier.joblib")
x, y = read_dataset(args.xlsx)
if not 0 <= args.index < len(x):
    raise SystemExit(f"index必须在0到{len(x)-1}之间")
row = x.iloc[args.index:args.index + 1]
true_label = str(y.iloc[args.index])
pred_id = int(artifact["model"].predict(row)[0])
pred_label = str(artifact["label_encoder"].inverse_transform([pred_id])[0])
probabilities = artifact["model"].predict_proba(row)[0]
confidence = float(np.max(probabilities))

print("=" * 42)
print("           MQ气体模型单条测试结果")
print("=" * 42)
print(f"真实类别：{true_label}")
print(f"预测类别：{pred_label}")
print(f"识别置信度：{confidence:.1%}")
print("传感器数据：")
for sensor, value in row.iloc[0].items():
    print(f"  {sensor}: {float(value):.4f}")
print(f"分类结果：{'正确' if true_label == pred_label else '错误'}")
print("=" * 42)
