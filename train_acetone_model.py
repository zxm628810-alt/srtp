"""Train a dedicated acetone detector and acetone concentration regressor."""
from pathlib import Path
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from train_uci_gas_model import load

ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "uci_gas" / "unzipped")
args = ap.parse_args()
x, gas, ppm = load(args.data_dir)
is_acetone = (gas == 1).astype(int)
xtr, xte, ytr, yte = train_test_split(x, is_acetone, test_size=.2, random_state=42, stratify=is_acetone)
clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
clf.fit(xtr, ytr); pred = clf.predict(xte)
print(classification_report(yte, pred, target_names=["非丙酮", "丙酮"], zero_division=0))

mask = gas == 1
xa, pa = x[mask], ppm[mask]
xa_tr, xa_te, pa_tr, pa_te = train_test_split(xa, pa, test_size=.2, random_state=42)
reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
reg.fit(xa_tr, pa_tr); pa_pred = reg.predict(xa_te)
print(f"丙酮浓度 MAE: {mean_absolute_error(pa_te, pa_pred):.3f} ppm")
print(f"丙酮浓度 R2: {r2_score(pa_te, pa_pred):.3f}")
joblib.dump({"classifier": clf, "regressor": reg}, "acetone_model.joblib")
print("模型已保存: acetone_model.joblib")
