"""Convert UCI batch*.dat files into one reproducible CSV dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# Verified against the official batch distributions; do not use the display-order
# table as a raw-class-ID mapping.
GASES = {1: "Acetone", 2: "Acetaldehyde", 3: "Ethanol", 4: "Ethylene", 5: "Ammonia", 6: "Toluene"}

def parse_batch(path: Path, batch_id: int) -> list[dict]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts or ";" not in parts[0]:
            continue
        try:
            class_id, concentration = parts[0].split(";", 1)
            features = [float(x.split(":", 1)[1]) for x in parts[1:129]]
            if len(features) != 128:
                continue
            row = {"batch_id": batch_id, "gas_class": int(class_id),
                   "gas_name": GASES[int(class_id)], "concentration_ppm": float(concentration)}
            row.update({f"feature_{i+1}": value for i, value in enumerate(features)})
            rows.append(row)
        except (ValueError, IndexError):
            continue
    return rows

def main(data_dir: Path, output: Path) -> None:
    rows = []
    for batch_id in range(1, 11):
        path = data_dir / f"batch{batch_id}.dat"
        if not path.exists():
            raise FileNotFoundError(f"缺少文件: {path}")
        rows.extend(parse_batch(path, batch_id))
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"已生成: {output.resolve()}")
    print(f"样本数: {len(df)}; 字段数: {len(df.columns)}")
    print(df.groupby(["batch_id", "gas_name"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "uci_gas" / "unzipped")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "all_batches.csv")
    args = ap.parse_args(); main(args.data_dir, args.output)
