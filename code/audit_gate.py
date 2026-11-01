"""Data audit gate: cross-pond duplication check + data manifest. Run BEFORE any experiment.

Usage:  python audit_gate.py [--exclude 1] [--threshold 0.05]
  * Table of pairwise overlap among candidate ponds (5-min grid, as used for forecasting):
      ts_do  = fraction of pond i's rows whose timestamp AND DO value also occur in pond j
      allvar = fraction of pond i's rows whose six variables are all identical to a row of pond j
  * Gate: exits with status 1 if any pair of INCLUDED ponds has ts_do or allvar above the threshold.
  * Writes results/audit.json and results/data_manifest.json (SHA-256 of every raw CSV).
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import lopo_forecast as F

warnings.filterwarnings("ignore")
CAND = [1, 2, 3, 4, 6, 9]
CSV = Path(__file__).resolve().parents[1] / "data_raw" / "csv"


def manifest():
    out = {}
    for f in sorted(CSV.glob("*.csv")):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[f.name] = {"sha256": h.hexdigest(), "bytes": f.stat().st_size}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=0.05)
    a = ap.parse_args()
    S = {p: F.pond_series(p).dropna() for p in CAND}
    keys = {p: set(map(tuple, np.round(S[p].values, 4))) for p in CAND}
    ts_do, allvar = {}, {}
    for p in CAND:
        ts_do[p], allvar[p] = {}, {}
        vals = np.round(S[p].values, 4)
        for q in CAND:
            if q == p:
                continue
            j = S[p][["Dissolved Oxygen(g/ml)"]].join(S[q][["Dissolved Oxygen(g/ml)"]], rsuffix="_q", how="inner")
            ts_do[p][q] = float(np.isclose(j.iloc[:, 0], j.iloc[:, 1]).sum() / len(S[p]))
            allvar[p][q] = float(np.mean([tuple(x) in keys[q] for x in vals]))
    inc = [p for p in CAND if p not in a.exclude]
    bad = [(p, q, ts_do[p][q], allvar[p][q]) for p in inc for q in inc if p != q and max(ts_do[p][q], allvar[p][q]) > a.threshold]
    res = {"ponds": CAND, "rows": {p: int(len(S[p])) for p in CAND}, "ts_do": ts_do, "allvar": allvar, "excluded": a.exclude,
           "threshold": a.threshold, "violations": [{"i": p, "j": q, "ts_do": x, "allvar": y} for p, q, x, y in bad],
           "gate": "FAIL" if bad else "PASS"}
    out = Path(F.OUT)
    (out / "audit.json" if not a.exclude else out / "audit_after_exclusion.json").write_text(json.dumps(res, indent=1))
    (out / "data_manifest.json").write_text(json.dumps(manifest(), indent=1))
    print("included ponds:", inc, "| threshold", a.threshold)
    for p, q, x, y in bad:
        print(f"  VIOLATION pond {p} vs {q}: same timestamp+DO {100 * x:.1f}%, all six variables {100 * y:.1f}%")
    print("GATE:", res["gate"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
