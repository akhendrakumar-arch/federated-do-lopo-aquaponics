"""External check of the forecasting protocol on an independent public dataset.

Montaria/Monteria (Colombia) tilapia pond IoT data, Mendeley Data DOI 10.17632/3g2b4sh65m.1
(CC BY 4.0): hourly temperature, DO (mg/L), pH, turbidity, Jan-Jun 2024, one series.
Protocol: 24 h input window, horizons 1/3/6 h, chronological 70/15/15 split,
train-only clipping/scaling, persistence vs GBM vs GRU, 5 seeds.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats

import lopo_forecast as F

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
COLS = ["Temperature (°C)", "Dissolved Oxygen (mg/L)", "pH", "Turbidity (NTU)"]
DO = 1
WIN, HORIZ, SEEDS = 24, [1, 3, 6], [0, 1, 2, 3, 4]


def main():
    df = pd.read_excel(ROOT / "data_raw" / "monteria" / "monteria.xlsx")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    g = df.set_index("Datetime")[COLS].sort_index().resample("1h").mean()
    a = g.values.astype(np.float32)
    ok = ~np.isnan(a).any(1)
    X, Y = [], []
    for i in range(WIN - 1, len(a) - max(HORIZ)):
        if ok[i - WIN + 1:i + 1].all() and all(ok[i + h] for h in HORIZ):
            X.append(a[i - WIN + 1:i + 1]); Y.append([a[i + h, DO] for h in HORIZ])
    X, Y = np.array(X), np.array(Y, dtype=np.float32)
    n = len(X)
    i1, i2 = int(.70 * n), int(.85 * n)
    tr, va, te = (X[:i1], Y[:i1]), (X[i1 + max(HORIZ):i2], Y[i1 + max(HORIZ):i2]), (X[i2 + max(HORIZ):], Y[i2 + max(HORIZ):])
    flat = tr[0].reshape(-1, len(COLS))
    lo, hi = np.percentile(flat, .5, 0), np.percentile(flat, 99.5, 0)
    mu, sd = np.clip(flat, lo, hi).mean(0), np.clip(flat, lo, hi).std(0) + 1e-6
    norm = lambda Z: ((np.clip(Z, lo, hi) - mu) / sd).astype(np.float32)
    tgt = lambda Z: ((np.clip(Z, lo[DO], hi[DO]) - mu[DO]) / sd[DO]).astype(np.float32)
    back = lambda z: z * sd[DO] + mu[DO]
    Ntr, Nva, Nte = norm(tr[0]), norm(va[0]), norm(te[0])
    Ttr, Tva = tgt(tr[1]), tgt(va[1])
    F.HORIZ = HORIZ; F.DO = DO  # reuse Net/fit_nn with this dataset's layout
    import lopo_forecast
    lopo_forecast.HORIZ, lopo_forecast.DO = HORIZ, DO
    res = {"n": {"train": len(tr[0]), "val": len(va[0]), "test": len(te[0])}, "horizons_h": HORIZ}
    res["Persistence"] = F.metrics(te[1], np.repeat(np.clip(te[0], lo, hi)[:, -1, DO:DO + 1], len(HORIZ), 1))
    # lag-1 autocorrelation of DO: sanity check on how predictable / smooth the series is
    do = g[COLS[DO]].dropna().values
    res["do_lag1_autocorr"] = float(np.corrcoef(do[:-1], do[1:])[0, 1])
    res["do_std"] = float(do.std())
    S = lambda Z: np.concatenate([Z[:, -1], Z.mean(1), Z.std(1), Z[:, -1] - Z[:, 0], Z[:, -1] - Z[:, -4]], 1)
    gb = np.zeros((len(Nte), len(HORIZ)))
    for j in range(len(HORIZ)):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=.08, random_state=0)
        m.fit(S(Ntr), Ttr[:, j] - Ntr[:, -1, DO])
        gb[:, j] = m.predict(S(Nte)) + Nte[:, -1, DO]
    res["GBM"] = F.metrics(te[1], back(gb))
    runs = []
    for s in SEEDS:
        m = F.fit_nn("GRU", Ntr, Ttr, Nva, Tva, s)
        runs.append(F.metrics(te[1], back(F.predict(m, Nte))))
    res["GRU"] = {k: np.mean([r[k] for r in runs], 0).tolist() for k in runs[0]}
    res["GRU_rmse_seed_std"] = np.std([r["rmse"] for r in runs], 0).tolist()
    res["GRU_rmse_ci95"] = (stats.t.ppf(.975, len(SEEDS) - 1) * np.std([r["rmse"] for r in runs], 0, ddof=1) / np.sqrt(len(SEEDS))).tolist()
    (F.OUT / "external_monteria.json").write_text(json.dumps(res, indent=2))
    for k in ("Persistence", "GBM", "GRU"):
        print(k, "RMSE", np.round(res[k]["rmse"], 3), "R2", np.round(res[k]["r2"], 3))
    print("GRU seed CI95", np.round(res["GRU_rmse_ci95"], 3), "| DO lag-1 autocorr", round(res["do_lag1_autocorr"], 3), "std", round(res["do_std"], 3))


if __name__ == "__main__":
    main()
