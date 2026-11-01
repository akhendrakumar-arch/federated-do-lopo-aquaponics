"""Dataset summary table + example forecast trace (for paper figures).

dataset_summary.json : per-pond rows, period, windows, DO stats, weight/length ratio.
example_pred.json    : held-out pond, 1 h horizon, actual vs GRU vs persistence over a
                       contiguous segment, with a TargetInit conformal interval.
"""
import json
import warnings

import numpy as np
import pandas as pd

import lopo_classification as C
import lopo_forecast as F

warnings.filterwarnings("ignore")
DO, WIN, HORIZ = F.DO, F.WIN, F.HORIZ
HELD, J = 2, 1  # example pond, horizon index (1 h)


def windows_t(g):
    a, idx = g.values, g.index
    ok = ~np.isnan(a).any(axis=1)
    X, Y, T = [], [], []
    for i in range(WIN - 1, len(a) - max(HORIZ)):
        if ok[i - WIN + 1:i + 1].all() and all(ok[i + h] for h in HORIZ):
            X.append(a[i - WIN + 1:i + 1]); Y.append([a[i + h, DO] for h in HORIZ]); T.append(idx[i + HORIZ[J]])
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(T)


def summary():
    ratios = json.load(open(F.OUT / "lopo_classification.json"))["_meta"]
    out = {}
    for p in C.PONDS:
        d = C.load(p)
        do = pd.to_numeric(d["Dissolved Oxygen(g/ml)"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        row = {"rows": int(len(d)), "do_q": [float(x) for x in do.quantile([.25, .5, .75])],
               "ratio": ratios["ratios"][str(p)], "label": ratios["labels"][str(p)]}
        if p in F.PONDS:
            g = F.pond_series(p)
            X, _, _ = windows_t(g)
            row.update(windows=int(len(X)), start=str(g.index.min().date()), end=str(g.index.max().date()))
        out[p] = row
    (F.OUT / "dataset_summary.json").write_text(json.dumps(out, indent=2))
    print({p: (v["rows"], v.get("windows")) for p, v in out.items()})


def example():
    raw = {p: windows_t(F.pond_series(p)) for p in F.PONDS}
    tr, va = [], []
    for p in F.PONDS:
        if p == HELD:
            continue
        X, Y, _ = raw[p]
        k = int(len(X) * 0.85)
        tr.append((X[:k], Y[:k])); va.append((X[k:], Y[k:]))
    Xtr = np.concatenate([a for a, _ in tr]); Ytr = np.concatenate([b for _, b in tr])
    Xva = np.concatenate([a for a, _ in va]); Yva = np.concatenate([b for _, b in va])
    rng = np.random.default_rng(0)
    if len(Xtr) > F.MAXTRAIN:
        s = rng.choice(len(Xtr), F.MAXTRAIN, replace=False); Xtr, Ytr = Xtr[s], Ytr[s]
    Xh, Yh, Th = raw[HELD]
    flat = Xtr.reshape(-1, Xtr.shape[2])
    lo, hi = np.percentile(flat, .5, 0), np.percentile(flat, 99.5, 0)
    clip = lambda X: np.clip(X, lo, hi)
    mu, sd = clip(flat).mean(0), clip(flat).std(0) + 1e-6
    norm = lambda X: ((clip(X) - mu) / sd).astype(np.float32)
    tgt = lambda Y: ((np.clip(Y, lo[DO], hi[DO]) - mu[DO]) / sd[DO]).astype(np.float32)
    back = lambda z: z * sd[DO] + mu[DO]
    preds = []
    for s in F.SEEDS:
        m = F.fit_nn("GRU", norm(Xtr), tgt(Ytr), norm(Xva), tgt(Yva), s)
        preds.append(back(F.predict(m, norm(Xh))))
    ph = np.mean(preds, 0)[:, J]
    y = Yh[:, J]
    pers = clip(Xh)[:, -1, DO]
    a = int(len(Xh) * 0.15)
    q = float(np.quantile(np.abs(y[:a] - ph[:a]), 0.9))
    # longest contiguous run of windows (5-min steps) after the adaptation segment, take 3 days from its middle
    step = np.diff(Th.astype("datetime64[s]").astype(np.int64)) == 300
    best, start, cur = (0, 0), a + 48, a + 48
    for i in range(a + 48, len(step)):
        if step[i]:
            continue
        if i - cur > best[1] - best[0]:
            best = (cur, i)
        cur = i + 1
    if len(step) - cur > best[1] - best[0]:
        best = (cur, len(step))
    L = 3 * 24 * 12
    # pick the 3-day stretch (stride 6 h) whose observed-DO spread is closest to the pond's overall spread,
    # so the example is not a flat-lined sensor stretch
    target = float(np.std(y))
    L = 36 * 12  # 36 h; 3-day stretches are rare or flat in this pond
    cands = [s for s in range(int(len(Xh) * 0.15) + 48, len(y) - L, 36) if step[s:s + L - 1].all()]
    stds = {s: float(np.std(y[s:s + L])) for s in cands}
    ok = sorted((s for s in cands if 1.5 < stds[s] < 6 and y[s:s + L].max() < 25), key=lambda s: stds[s])
    s0 = ok[len(ok) // 2] if ok else max(cands, key=lambda s: stds[s])
    print("candidates", len(cands), "typical", len(ok), "chosen std", round(stds[s0], 2))
    sl = slice(s0, min(best[1], s0 + L))
    t = [str(x)[:16] for x in Th[sl]]
    seg = lambda v: [round(float(x), 3) for x in v[sl]]
    json.dump({"pond": HELD, "horizon_min": HORIZ[J] * 5, "time": t, "actual": seg(y), "gru": seg(ph), "persistence": seg(pers),
               "q90_target_init": q, "n": len(t)}, open(F.OUT / "example_pred.json", "w"))
    print("example", len(t), t[0], t[-1], "q", round(q, 2),
          "seg RMSE gru/pers", np.sqrt(np.mean((np.array(seg(ph)) - np.array(seg(y))) ** 2)).round(2),
          np.sqrt(np.mean((np.array(seg(pers)) - np.array(seg(y))) ** 2)).round(2))


if __name__ == "__main__":
    summary()
    example()
