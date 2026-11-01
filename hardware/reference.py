"""float64 numpy reference of the hardware protocol (what the devices must reproduce).

Run:  python reference.py      -> reference_results.json
Also compares with the paper's pooled LOPO Linear AR (results/lopo_baselines.json); the two
differ slightly only because the protocol derives clip bounds from a fixed-bin histogram.
"""
import json
from pathlib import Path

import numpy as np

from hwcommon import BW, FOLDS, NB, percentile_from_hist, solve_ridge

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load():
    meta = json.loads((DATA / "meta.json").read_text())
    arr = {p: np.fromfile(DATA / f"pond{p}.f32", dtype="<f4").reshape(-1, 15) for p in FOLDS}
    return meta, arr


def run():
    meta, arr = load()
    out = {}
    for h in FOLDS:
        tr = {p: arr[p][:meta[str(p)]["n_train"]].astype(np.float64) for p in FOLDS if p != h}
        hist = np.zeros(NB + 1)
        for a in tr.values():
            v = a[:, :12].ravel()
            hist += np.bincount(np.where(v < 0, 0, np.minimum((v / BW).astype(int), NB)), minlength=NB + 1)
        lo, hi = percentile_from_hist(hist.tolist(), 0.5), percentile_from_hist(hist.tolist(), 99.5)
        n = s1 = s2 = 0.0
        for a in tr.values():
            c = np.clip(a[:, :12], lo, hi)
            n += c.size; s1 += c.sum(); s2 += (c * c).sum()
        mu = s1 / n
        sd = float(np.sqrt(max(s2 / n - mu * mu, 0.0))) + 1e-6
        A, B = np.zeros((13, 13)), np.zeros((13, 3))
        for a in tr.values():
            xn = (np.clip(a[:, :12], lo, hi) - mu) / sd
            D = np.concatenate([xn, np.ones((len(a), 1))], 1)
            T = (np.clip(a[:, 12:], lo, hi) - mu) / sd - xn[:, 11:12]
            A += D.T @ D; B += D.T @ T
        W = np.array(solve_ridge(A.ravel().tolist(), B.ravel().tolist())).reshape(13, 3)
        te = arr[h].astype(np.float64)
        xn = (np.clip(te[:, :12], lo, hi) - mu) / sd
        D = np.concatenate([xn, np.ones((len(te), 1))], 1)
        pred = (xn[:, 11:12] + D @ W) * sd + mu
        pers = np.clip(te[:, 11:12], lo, hi)
        y = te[:, 12:]
        out[h] = {"lo": lo, "hi": hi, "mu": mu, "sd": sd, "W": W.ravel().tolist(), "n": int(len(te)),
                  "rmse": np.sqrt(((y - pred) ** 2).mean(0)).tolist(), "rmse_pers": np.sqrt(((y - pers) ** 2).mean(0)).tolist(),
                  "sse": ((y - pred) ** 2).sum(0).tolist(), "sse_pers": ((y - pers) ** 2).sum(0).tolist(),
                  "sum_pred": pred.sum(0).tolist(), "first": pred[:5].ravel().tolist()}
    (HERE / "reference_results.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    out = run()
    base = json.loads((HERE.parent / "results" / "lopo_baselines.json").read_text())["folds"]
    print("pond | n | RMSE ref (15m,1h,3h) | persistence | paper pooled Linear AR (all windows)")
    for h, r in out.items():
        print(h, r["n"], np.round(r["rmse"], 3), np.round(r["rmse_pers"], 3), np.round(base[str(h)]["Linear AR"]["rmse"], 3))
