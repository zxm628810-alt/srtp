"""Human-readable single-sample test for the UCI gas model."""
from pathlib import Path
import argparse
import joblib
import numpy as np
from train_uci_gas_model import load, GASES

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "uci_gas" / "unzipped")
ap.add_argument("--index", type=int, default=0, help="样本编号，从0开始")
args = ap.parse_args()

artifact = joblib.load("uci_gas_model.joblib")
x, y, ppm = load(args.data_dir)
if not 0 <= args.index < len(x):
    raise SystemExit(f"index必须在0到{len(x)-1}之间")
row = x[args.index:args.index + 1]
true_id, true_ppm = int(y[args.index]), float(ppm[args.index])
pred_id = int(artifact["classifier"].predict(row)[0])
confidence = float(np.max(artifact["classifier"].predict_proba(row)[0]))
pred_ppm = float(artifact["regressor"].predict(row)[0])

print("=" * 42)
print("          UCI气体模型单条测试结果")
print("=" * 42)
print(f"真实气体：{GASES[true_id]}")
print(f"预测气体：{GASES[pred_id]}")
print(f"识别置信度：{confidence:.1%}")
print(f"真实浓度：{true_ppm:.2f} ppm")
print(f"预测浓度：{pred_ppm:.2f} ppm")
print(f"浓度误差：{abs(true_ppm-pred_ppm):.2f} ppm")
print(f"分类结果：{'正确' if true_id == pred_id else '错误'}")
print("=" * 42)
