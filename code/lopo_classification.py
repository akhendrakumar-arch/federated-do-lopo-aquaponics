"""Leave-one-pond-out (LOPO) classification of pond condition from water quality.

Fixes the thesis loader defects: each pond is read from its own CSV (no pond is
duplicated), and no row from the held-out pond is ever seen in training. Scaling/
imputation are fit on training ponds only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data_raw" / "csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

PONDS = [1, 2, 3, 4, 6, 7, 8, 9]  # no IoTPond5 file exists in the Kaggle release
FEATS = ["Temperature (C)", "Turbidity (NTU)", "Dissolved Oxygen(g/ml)", "PH",
         "Ammonia(g/ml)", "Nitrate(g/ml)"]
CLASSES = ["Worst", "Bad", "Good", "Best"]


KEYS = [("temp", "Temperature (C)"), ("turb", "Turbidity (NTU)"), ("oxygen", "Dissolved Oxygen(g/ml)"),
        ("ph", "PH"), ("ammon", "Ammonia(g/ml)"), ("nitrate", "Nitrate(g/ml)"),
        ("length", "Fish_Length (cm)"), ("lenght", "Fish_Length (cm)"), ("weight", "Fish_Weight (g)")]


def load(p):
    """Read one pond CSV and map its (inconsistent) headers to canonical names."""
    for name in (f"IoTPond{p}.csv", f"IoTpond{p}.csv"):
        f = CSV / name
        if f.exists():
            df = pd.read_csv(f)
            ren = {}
            for c in df.columns:
                lc = str(c).lower().replace(" ", "")
                for k, canon in KEYS:
                    if (lc == "ph") if k == "ph" else (k in lc):
                        if canon not in ren.values():
                            ren[c] = canon
                        break
            return df.rename(columns=ren)
    return None


def main():
    frames, ratios = [], {}
    for p in PONDS:
        df = load(p)
        if df is None:
            print(f"pond {p}: missing, skipped")
            continue
        df = df.copy()
        for c in FEATS + ["Fish_Length (cm)", "Fish_Weight (g)"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)  # NaNs are imputed with train-only medians below
        ratios[p] = float((df["Fish_Weight (g)"] / df["Fish_Length (cm)"]).mean())
        df["pond"] = p
        frames.append(df[FEATS + ["pond"]])
        print(f"pond {p}: {len(df)} rows, W/L ratio {ratios[p]:.4f}")
    ponds = sorted(ratios)
    order = sorted(ponds, key=lambda p: ratios[p])
    q = np.array_split(order, 4)  # rank-based quartiles of pond W/L ratio
    label = {int(p): i for i, grp in enumerate(q) for p in grp}
    print("pond labels:", {p: CLASSES[label[p]] for p in sorted(label)})
    data = pd.concat(frames, ignore_index=True)
    data["y"] = data["pond"].map(label)

    models = {
        "RF": lambda: RandomForestClassifier(200, n_jobs=-1, random_state=0),
        "DT": lambda: DecisionTreeClassifier(max_depth=12, random_state=0),
        "HGB": lambda: HistGradientBoostingClassifier(random_state=0),
    }
    res = {}
    for name, mk in models.items():
        row_acc, pond_hits, folds = [], [], {}
        for held in sorted(label):
            tr, te = data[data.pond != held], data[data.pond == held]
            tr = tr.sample(min(len(tr), 150000), random_state=0)  # cap for speed
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), mk())
            pipe.fit(tr[FEATS], tr["y"])
            pred = pipe.predict(te[FEATS])
            acc = accuracy_score(te["y"], pred)
            vote = np.bincount(pred, minlength=4).argmax()
            folds[held] = {"row_acc": acc, "vote": CLASSES[vote], "true": CLASSES[label[held]]}
            row_acc.append(acc)
            pond_hits.append(int(vote == label[held]))
            print(f"{name} hold-out pond {held}: row acc {acc:.3f} vote {CLASSES[vote]} (true {CLASSES[label[held]]})")
        res[name] = {"mean_row_acc": float(np.mean(row_acc)), "std_row_acc": float(np.std(row_acc)),
                     "pond_level_acc": float(np.mean(pond_hits)), "folds": folds}
        print(f"== {name}: mean row acc {np.mean(row_acc):.3f} +/- {np.std(row_acc):.3f}, pond-level {np.mean(pond_hits):.2f}")

    # chance baseline for reference
    res["_meta"] = {"ratios": ratios, "labels": {str(k): CLASSES[v] for k, v in label.items()}, "chance": 0.25}
    (OUT / "lopo_classification.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    sys.exit(main())
