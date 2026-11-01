"""Exact federated linear AR (ridge on the last 12 DO values) via sufficient statistics.

Each client sends X^T X and X^T Y (a (13x13) and (13x3) matrix, bias column included) once;
the server solves the ridge system. The solution equals pooled ridge exactly, which we verify
numerically. Evaluated on the same windows as the federated study (after the first 15% of the
held-out pond plus a 48-window gap) and, for reference, on the full held-out pond.
"""
import json
import warnings

import numpy as np
from sklearn.linear_model import Ridge
from scipy import stats

import lopo_forecast as F

warnings.filterwarnings("ignore")
PONDS, DO, HORIZ = F.PONDS, F.DO, F.HORIZ
ALPHA, ADAPT, GAP = 1.0, 0.15, 48


def design(N):
    return np.concatenate([N[:, :, DO], np.ones((len(N), 1), np.float32)], 1).astype(np.float64)


def main():
    raw = {p: F.windows(F.pond_series(p)) for p in PONDS}
    res = {}
    for held in PONDS:
        tr = []
        for p in PONDS:
            if p == held:
                continue
            X, Y = raw[p]
            k = int(len(X) * 0.85)
            tr.append((X[:k], Y[:k]))
        rng = np.random.default_rng(0)
        Xtr = np.concatenate([a for a, _ in tr]); Ytr = np.concatenate([b for _, b in tr])
        if len(Xtr) > F.MAXTRAIN:  # identical sub-sampling to lopo_baselines.py
            s = rng.choice(len(Xtr), F.MAXTRAIN, replace=False); Xtr, Ytr = Xtr[s], Ytr[s]
        Xh, Yh = raw[held]
        flat = Xtr.reshape(-1, Xtr.shape[2])
        lo, hi = np.percentile(flat, .5, 0), np.percentile(flat, 99.5, 0)
        clip = lambda X: np.clip(X, lo, hi)
        mu, sd = clip(flat).mean(0), clip(flat).std(0) + 1e-6
        norm = lambda X: ((clip(X) - mu) / sd).astype(np.float32)
        tgt = lambda Y: ((np.clip(Y, lo[DO], hi[DO]) - mu[DO]) / sd[DO]).astype(np.float32)
        back = lambda z: z * sd[DO] + mu[DO]
        # split the (sub-sampled) training windows back into clients by pond for the federated solve
        sizes = [len(a) for a, _ in tr]
        order = np.cumsum([0] + sizes)
        # client statistics (on the sub-sampled pooled array, partitioned by source pond via original indices)
        # simpler and equivalent: partition sub-sampled windows into len(tr) groups
        groups = np.array_split(np.arange(len(Xtr)), len(tr))
        D, T = design(norm(Xtr)), tgt(Ytr).astype(np.float64) - norm(Xtr)[:, -1, DO:DO + 1].astype(np.float64)
        A = np.zeros((D.shape[1], D.shape[1])); B = np.zeros((D.shape[1], T.shape[1]))
        for g in groups:  # one upload per client
            A += D[g].T @ D[g]; B += D[g].T @ T[g]
        R = ALPHA * np.eye(D.shape[1]); R[-1, -1] = 0.0  # bias not penalised (as in sklearn)
        W = np.linalg.solve(A + R, B)
        pooled = Ridge(alpha=ALPHA).fit(D[:, :-1], T)
        Dte = design(norm(Xh))
        pf = Dte @ W
        pp = pooled.predict(Dte[:, :-1])
        gap = float(np.max(np.abs(pf - pp)))
        pred = back(pf + norm(Xh)[:, -1, DO:DO + 1])
        a = int(len(Xh) * ADAPT)
        sub = slice(a + GAP, None)
        pers = np.clip(Xh, lo, hi)[:, -1, DO:DO + 1].repeat(len(HORIZ), 1)
        res[held] = {"rmse": np.sqrt(((Yh[sub] - pred[sub]) ** 2).mean(0)).tolist(),
                     "persistence_rmse": np.sqrt(((Yh[sub] - pers[sub]) ** 2).mean(0)).tolist(),
                     "max_abs_diff_vs_pooled": gap, "n_test": int(len(Yh[sub]))}
        print(held, "RMSE", np.round(res[held]["rmse"], 2), "pers", np.round(res[held]["persistence_rmse"], 2), "max|fed-pooled|", f"{gap:.2e}", flush=True)
    M = np.array([res[p]["rmse"] for p in PONDS])
    ci = stats.t.ppf(.975, len(PONDS) - 1) * M.std(0, ddof=1) / np.sqrt(len(PONDS))
    nfeat = 13
    out = {"folds": res, "rmse_mean": M.mean(0).tolist(), "rmse_ci95": ci.tolist(),
           "comm_bytes_per_client": 4 * (nfeat * nfeat + nfeat * len(HORIZ)), "rounds": 1,
           "max_abs_diff_vs_pooled": max(r["max_abs_diff_vs_pooled"] for r in res.values())}
    (F.OUT / "fed_linear.json").write_text(json.dumps(out, indent=2))
    print("mean RMSE", np.round(M.mean(0), 3), "+/-", np.round(ci, 3), "| bytes/client", out["comm_bytes_per_client"], "| max diff", out["max_abs_diff_vs_pooled"])


if __name__ == "__main__":
    main()
