"""Leave-one-pond-out multi-horizon dissolved-oxygen forecasting.

For each held-out pond the models train only on the other ponds. Clipping bounds,
imputation values and scaling are learned from training ponds only. Early stopping
and conformal calibration use the chronological last 15% of each training pond.

Models: persistence, HistGradientBoosting (on window summary features), GRU, LSTM.
Horizons: 15 min, 1 h, 3 h (5-min grid, 60-min input window).
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

import lopo_classification as L

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
OUT = L.OUT
# Ponds with clean timestamps. Pond 1 is excluded: 92% of its rows share timestamp and DO value
# with Pond 2 (73% match on all six variables), so it would leak across LOPO folds.
PONDS = [2, 3, 4, 6, 9]
FEATS = L.FEATS
DO = FEATS.index("Dissolved Oxygen(g/ml)")
WIN, HORIZ = 12, [3, 12, 36]  # steps of 5 min
SEEDS = [0, 1, 2]
EPOCHS, BATCH, LR = 25, 512, 2e-3
MAXTRAIN = 60000


def pond_series(p):
    d = L.load(p)
    t = d["created_at"].astype(str).str.replace(r"\s*CES?T$", "", regex=True)
    d["t"] = pd.to_datetime(t, errors="coerce", utc=True)
    d = d.dropna(subset=["t"])
    for c in FEATS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.set_index("t")[FEATS].sort_index().resample("5min").median()


def windows(g):
    """All windows whose inputs and targets are fully observed. Returns X, Y, last-DO."""
    a = g.values
    n, hmax = len(a), max(HORIZ)
    ok = ~np.isnan(a).any(axis=1)
    X, Y = [], []
    for i in range(WIN - 1, n - hmax):
        if ok[i - WIN + 1:i + 1].all() and all(ok[i + h] for h in HORIZ):
            X.append(a[i - WIN + 1:i + 1])
            Y.append([a[i + h, DO] for h in HORIZ])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def summary(X):
    last, d = X[:, -1], X[:, -1] - X[:, 0]
    return np.concatenate([last, X.mean(1), X.std(1), d, X[:, -1] - X[:, -4]], axis=1)


class Net(nn.Module):
    def __init__(self, kind, nf, nh=64):
        super().__init__()
        self.rnn = (nn.GRU if kind == "GRU" else nn.LSTM)(nf, nh, batch_first=True)
        self.head = nn.Sequential(nn.Linear(nh, 64), nn.ReLU(), nn.Linear(64, len(HORIZ)))

    def forward(self, x):
        o, _ = self.rnn(x)
        return x[:, -1, DO:DO + 1] + self.head(o[:, -1])  # residual on last DO


def fit_nn(kind, Xtr, Ytr, Xva, Yva, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Net(kind, Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    Xt, Yt, Xv, Yv = map(torch.tensor, (Xtr, Ytr, Xva, Yva))
    best, bad, state = 1e9, 0, None
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = nn.functional.smooth_l1_loss(m(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(m(Xv), Yv).item()
        if v < best:
            best, bad, state = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    m.load_state_dict(state)
    return m.eval()


def predict(m, X):
    with torch.no_grad():
        return m(torch.tensor(X)).numpy()


def metrics(y, p):
    e = y - p
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return {"rmse": np.sqrt((e ** 2).mean(0)).tolist(), "mae": np.abs(e).mean(0).tolist(),
            "r2": (1 - (e ** 2).sum(0) / ss).tolist()}


def main():
    t0 = time.time()
    raw = {p: windows(pond_series(p)) for p in PONDS}
    for p, (X, Y) in raw.items():
        print(f"pond {p}: {len(X)} windows", flush=True)
    results = {}
    for held in PONDS:
        # chronological split inside each training pond
        tr, va = [], []
        for p in PONDS:
            if p == held:
                continue
            X, Y = raw[p]
            k = int(len(X) * 0.85)
            tr.append((X[:k], Y[:k]))
            va.append((X[k:], Y[k:]))
        Xtr = np.concatenate([a for a, _ in tr]); Ytr = np.concatenate([b for _, b in tr])
        Xva = np.concatenate([a for a, _ in va]); Yva = np.concatenate([b for _, b in va])
        rng = np.random.default_rng(0)
        if len(Xtr) > MAXTRAIN:
            sel = rng.choice(len(Xtr), MAXTRAIN, replace=False)
            Xtr, Ytr = Xtr[sel], Ytr[sel]
        Xte, Yte = raw[held]
        # train-only preprocessing: clip bounds + scaling per feature
        flat = Xtr.reshape(-1, Xtr.shape[2])
        lo, hi = np.percentile(flat, 0.5, axis=0), np.percentile(flat, 99.5, axis=0)
        clip = lambda X: np.clip(X, lo, hi)
        mu, sd = clip(flat).mean(0), clip(flat).std(0) + 1e-6
        norm = lambda X: ((clip(X) - mu) / sd).astype(np.float32)
        tgt = lambda Y: ((np.clip(Y, lo[DO], hi[DO]) - mu[DO]) / sd[DO]).astype(np.float32)
        Ntr, Nva, Nte = norm(Xtr), norm(Xva), norm(Xte)
        Ttr, Tva = tgt(Ytr), tgt(Yva)
        back = lambda z: z * sd[DO] + mu[DO]
        fold = {}
        # persistence (uses raw last observed DO; clipped for parity with the scaled models)
        fold["Persistence"] = metrics(Yte, np.repeat(clip(Xte)[:, -1, DO:DO + 1], len(HORIZ), 1))
        # gradient boosting on summary features, one model per horizon, predicting the change
        S = [summary(a) for a in (Ntr, Nva, Nte)]
        gp = np.zeros((len(Nte), len(HORIZ)))
        gva = np.zeros((len(Nva), len(HORIZ)))
        for j in range(len(HORIZ)):
            g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)
            g.fit(S[0], Ttr[:, j] - Ntr[:, -1, DO])
            gp[:, j] = g.predict(S[2]) + Nte[:, -1, DO]
            gva[:, j] = g.predict(S[1]) + Nva[:, -1, DO]
        fold["GBM"] = metrics(Yte, back(gp))
        for kind in ("GRU", "LSTM"):
            ps, pv, ms = [], [], []
            for s in SEEDS:
                m = fit_nn(kind, Ntr, Ttr, Nva, Tva, s)
                pt, pva = predict(m, Nte), predict(m, Nva)
                ps.append(back(pt)); pv.append(back(pva))
                ms.append(metrics(Yte, back(pt)))
            fold[kind] = {k: np.mean([x[k] for x in ms], 0).tolist() for k in ms[0]}
            fold[kind + "_seed_std_rmse"] = np.std([x["rmse"] for x in ms], 0).tolist()
            if kind == "GRU":  # split-conformal 90% interval from validation residuals
                ens_te, ens_va = np.mean(ps, 0), np.mean(pv, 0)
                q = np.quantile(np.abs(Yva - ens_va), 0.9, axis=0)
                fold["GRU_conformal"] = {"coverage": (np.abs(Yte - ens_te) <= q).mean(0).tolist(),
                                         "halfwidth": q.tolist(), "target": 0.9}
                fold["GRU_ens"] = metrics(Yte, ens_te)
                gru_model, gru_x = m, Nte[:1]
        fold["n_test"] = len(Xte)
        results[held] = fold
        print(f"held {held}: n={len(Xte)} RMSE(15m,1h,3h) " + "  ".join(
            f"{k}={np.round(fold[k]['rmse'], 2).tolist()}" for k in ("Persistence", "GBM", "GRU", "LSTM")),
            f"| {time.time() - t0:.0f}s", flush=True)

    # latency / size of the GRU on this CPU, batch of one window
    with torch.no_grad():
        for _ in range(20):
            gru_model(torch.tensor(gru_x))
        t = time.perf_counter()
        for _ in range(500):
            gru_model(torch.tensor(gru_x))
        lat = (time.perf_counter() - t) / 500 * 1e3
    params = sum(p.numel() for p in gru_model.parameters())

    # aggregate over held-out ponds: mean and 95% t-interval
    agg = {}
    for name in ("Persistence", "GBM", "GRU", "LSTM", "GRU_ens"):
        agg[name] = {}
        for met in ("rmse", "mae", "r2"):
            M = np.array([results[p][name][met] for p in PONDS])
            ci = stats.t.ppf(0.975, len(PONDS) - 1) * M.std(0, ddof=1) / np.sqrt(len(PONDS))
            agg[name][met] = {"mean": M.mean(0).tolist(), "ci95": ci.tolist()}
    skill = {}
    for name in ("GBM", "GRU", "LSTM"):
        S = np.array([1 - np.array(results[p][name]["rmse"]) / np.array(results[p]["Persistence"]["rmse"])
                      for p in PONDS])
        skill[name] = {"mean": S.mean(0).tolist(), "per_pond_min": S.min(0).tolist(),
                       "ponds_beating_persistence": (S > 0).sum(0).tolist()}
    cov = np.array([results[p]["GRU_conformal"]["coverage"] for p in PONDS])
    out = {"horizons_min": [h * 5 for h in HORIZ], "folds": results, "aggregate": agg, "rmse_skill_vs_persistence": skill,
           "conformal_coverage_mean": cov.mean(0).tolist(), "conformal_coverage_min": cov.min(0).tolist(),
           "gru_params": params, "gru_cpu_latency_ms": lat, "seeds": SEEDS, "ponds": PONDS}
    (OUT / "lopo_forecast.json").write_text(json.dumps(out, indent=2))
    print("done", f"{time.time() - t0:.0f}s; GRU params {params}, latency {lat:.2f} ms")
    for n in ("Persistence", "GBM", "GRU", "LSTM"):
        print(n, "RMSE", np.round(agg[n]["rmse"]["mean"], 3), "+/-", np.round(agg[n]["rmse"]["ci95"], 3))
    print("skill vs persistence", {k: np.round(v["mean"], 3).tolist() for k, v in skill.items()})
    print("conformal coverage", np.round(cov.mean(0), 3), "min", np.round(cov.min(0), 3))


if __name__ == "__main__":
    main()
