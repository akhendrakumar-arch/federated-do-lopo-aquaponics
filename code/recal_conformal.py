"""Conformal recalibration on the target pond (leave-one-pond-out GRU, 3-seed ensemble).

Schemes (all target 90% coverage, scored on the same held-out windows: after the
first 15% of the pond plus a 48-window gap):
  Source    - quantile of |residual| on validation windows of the training ponds
  TargetInit- quantile on the held-out pond's first 15% (labels observed once, then fixed)
  Rolling   - quantile of the most recent ROLL residuals whose targets are already
              known at forecast time (window j uses windows <= j - h - 1)
"""
import json
import time
import warnings

import numpy as np

import lopo_forecast as F

warnings.filterwarnings("ignore")
PONDS, HORIZ, DO, SEEDS = F.PONDS, F.HORIZ, F.DO, F.SEEDS
ALPHA, ADAPT, GAP, ROLL = 0.10, 0.15, 48, 500


def rolling_q(res, h):
    """res: |residual| array for one horizon over the whole pond in time order."""
    n, q = len(res), np.full(len(res), np.nan)
    for j in range(n):
        past = res[max(0, j - h - ROLL):max(0, j - h)]  # targets observed by time j
        if len(past) >= 50:
            q[j] = np.quantile(past, 1 - ALPHA)
    return q


def main():
    t0 = time.time()
    raw = {p: F.windows(F.pond_series(p)) for p in PONDS}
    out = {}
    for held in PONDS:
        tr, va = [], []
        for p in PONDS:
            if p == held:
                continue
            X, Y = raw[p]
            k = int(len(X) * 0.85)
            tr.append((X[:k], Y[:k])); va.append((X[k:], Y[k:]))
        Xtr = np.concatenate([a for a, _ in tr]); Ytr = np.concatenate([b for _, b in tr])
        Xva = np.concatenate([a for a, _ in va]); Yva = np.concatenate([b for _, b in va])
        rng = np.random.default_rng(0)
        if len(Xtr) > F.MAXTRAIN:
            s = rng.choice(len(Xtr), F.MAXTRAIN, replace=False); Xtr, Ytr = Xtr[s], Ytr[s]
        Xh, Yh = raw[held]
        flat = Xtr.reshape(-1, Xtr.shape[2])
        lo, hi = np.percentile(flat, 0.5, 0), np.percentile(flat, 99.5, 0)
        clip = lambda X: np.clip(X, lo, hi)
        mu, sd = clip(flat).mean(0), clip(flat).std(0) + 1e-6
        norm = lambda X: ((clip(X) - mu) / sd).astype(np.float32)
        tgt = lambda Y: ((np.clip(Y, lo[DO], hi[DO]) - mu[DO]) / sd[DO]).astype(np.float32)
        back = lambda z: z * sd[DO] + mu[DO]
        Ntr, Nva, Nh = norm(Xtr), norm(Xva), norm(Xh)
        ph, pv = [], []
        for s in SEEDS:
            m = F.fit_nn("GRU", Ntr, tgt(Ytr), Nva, tgt(Yva), s)
            ph.append(back(F.predict(m, Nh))); pv.append(back(F.predict(m, Nva)))
        ph, pv = np.mean(ph, 0), np.mean(pv, 0)
        res_h = np.abs(Yh - ph)
        a = int(len(Xh) * ADAPT)
        test = np.arange(a + GAP, len(Xh))
        qsrc = np.quantile(np.abs(Yva - pv), 1 - ALPHA, axis=0)
        qinit = np.quantile(res_h[:a], 1 - ALPHA, axis=0)
        fold = {"n_test": int(len(test))}
        for j, h in enumerate(HORIZ):
            qr = rolling_q(res_h[:, j], h)[test]
            ok = ~np.isnan(qr)
            r = res_h[test, j]
            fold[f"h{h * 5}"] = {
                "Source": {"coverage": float((r <= qsrc[j]).mean()), "width": float(2 * qsrc[j])},
                "TargetInit": {"coverage": float((r <= qinit[j]).mean()), "width": float(2 * qinit[j])},
                "Rolling": {"coverage": float((r[ok] <= qr[ok]).mean()), "width": float(2 * qr[ok].mean())}}
        out[held] = fold
        print(f"held {held}: " + "  ".join(
            f"h{h * 5}: " + "/".join(f"{fold[f'h{h * 5}'][k]['coverage']:.2f}" for k in ('Source', 'TargetInit', 'Rolling'))
            for h in HORIZ), f"| {time.time() - t0:.0f}s", flush=True)
    agg = {}
    for h in HORIZ:
        for k in ("Source", "TargetInit", "Rolling"):
            c = [out[p][f"h{h * 5}"][k]["coverage"] for p in PONDS]
            w = [out[p][f"h{h * 5}"][k]["width"] for p in PONDS]
            agg[f"h{h * 5}_{k}"] = {"cov_mean": float(np.mean(c)), "cov_min": float(np.min(c)), "cov_max": float(np.max(c)),
                                    "width_mean": float(np.mean(w))}
    (F.OUT / "recal_conformal.json").write_text(json.dumps({"folds": out, "aggregate": agg, "roll": ROLL, "alpha": ALPHA}, indent=2))
    for k, v in agg.items():
        print(k, {a: round(b, 3) for a, b in v.items()})


if __name__ == "__main__":
    main()
