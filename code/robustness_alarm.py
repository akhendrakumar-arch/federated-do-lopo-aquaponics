"""Robustness to stuck-sensor stretches and low-DO alarm detection (LOPO, clean 5 ponds).

Models: Persistence, Linear AR, GRU (3 seeds, mean prediction ensemble).
(a) RMSE on all test windows vs on 'active' windows (excluding windows whose 60 min input and
    1 h target DO are constant, i.e. suspected stuck sensor).
(b) Alarm: event = DO at the horizon falls below the 10th percentile of training-pond DO
    (threshold fixed from training ponds only); predicted event = forecast below the threshold.
    Precision / recall / F1 pooled over the held-out ponds, plus base rate.
"""
import json
import time
import warnings

import numpy as np
from sklearn.linear_model import Ridge

import lopo_forecast as F

warnings.filterwarnings("ignore")
PONDS, DO, HORIZ, SEEDS = F.PONDS, F.DO, F.HORIZ, F.SEEDS
Q = 10  # percentile defining the low-DO threshold


def main():
    t0 = time.time()
    raw = {p: F.windows(F.pond_series(p)) for p in PONDS}
    out = {"folds": {}, "q": Q}
    agg_ev = {m: {"tp": np.zeros(3), "fp": np.zeros(3), "fn": np.zeros(3)} for m in ("Persistence", "Linear AR", "GRU", "Persistence*", "Linear AR*", "GRU*")}
    base = np.zeros(3); ntot = 0
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
        lo, hi = np.percentile(flat, .5, 0), np.percentile(flat, 99.5, 0)
        clip = lambda X: np.clip(X, lo, hi)
        mu, sd = clip(flat).mean(0), clip(flat).std(0) + 1e-6
        norm = lambda X: ((clip(X) - mu) / sd).astype(np.float32)
        tgt = lambda Y: ((np.clip(Y, lo[DO], hi[DO]) - mu[DO]) / sd[DO]).astype(np.float32)
        back = lambda z: z * sd[DO] + mu[DO]
        Ntr, Nva, Nh = norm(Xtr), norm(Xva), norm(Xh)
        Ttr, Tva = tgt(Ytr), tgt(Yva)
        thr = float(np.percentile(Ytr[:, 1], Q))  # low-DO threshold from training-pond targets (1 h)
        last = lambda N: N[:, -1, DO]
        preds = {"Persistence": np.repeat(clip(Xh)[:, -1, DO:DO + 1], 3, 1)}
        ar = lambda N: N[:, :, DO]
        m = Ridge(alpha=1.0).fit(ar(Ntr), Ttr - last(Ntr)[:, None])
        preds["Linear AR"] = back(m.predict(ar(Nh)) + last(Nh)[:, None])
        gm = [F.fit_nn("GRU", Ntr, Ttr, Nva, Tva, s) for s in SEEDS]
        preds["GRU"] = np.mean([back(F.predict(g, Nh)) for g in gm], 0)
        vp = {"Persistence": np.repeat(clip(Xva)[:, -1, DO:DO + 1], 3, 1), "Linear AR": back(m.predict(ar(Nva)) + last(Nva)[:, None]),
              "GRU": np.mean([back(F.predict(g, Nva)) for g in gm], 0)}
        evv = Yva < thr
        cut = {}
        for n_, V in vp.items():  # alarm cutoff per horizon chosen on training-pond validation windows only
            cs = []
            for h_ in range(3):
                grid = np.percentile(V[:, h_], np.arange(2, 60, 1))
                f1s = [2 * ((evv[:, h_] & (V[:, h_] < c)).sum()) / max((V[:, h_] < c).sum() + evv[:, h_].sum(), 1) for c in grid]
                cs.append(float(grid[int(np.argmax(f1s))]))
            cut[n_] = np.array(cs)
            preds[n_ + "*"] = preds[n_]
        stuck = (Xh[:, :, DO].std(1) == 0) & (Yh[:, 1] == Xh[:, -1, DO])  # constant input and unchanged 1 h target
        fold = {"n": int(len(Yh)), "stuck_frac": float(stuck.mean()), "thr": thr}
        for name, P in preds.items():
            e = Yh - P
            rmse_all = np.sqrt((e ** 2).mean(0))
            rmse_act = np.sqrt((e[~stuck] ** 2).mean(0))
            ev = Yh < thr                      # true low-DO events at each horizon
            pr = P < (cut[name[:-1]] if name.endswith("*") else thr)
            tp, fp, fn = (ev & pr).sum(0), (~ev & pr).sum(0), (ev & ~pr).sum(0)
            fold[name] = {"rmse_all": rmse_all.tolist(), "rmse_active": rmse_act.tolist(),
                          "tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist()}
            agg_ev[name]["tp"] += tp; agg_ev[name]["fp"] += fp; agg_ev[name]["fn"] += fn
        base += (Yh < thr).sum(0); ntot += len(Yh)
        out["folds"][held] = fold
        print(f"held {held}: stuck {100 * fold['stuck_frac']:.1f}%  RMSE(1h) all/active  " + "  ".join(
            f"{n}={fold[n]['rmse_all'][1]:.2f}/{fold[n]['rmse_active'][1]:.2f}" for n in preds), f"| {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for name, a in agg_ev.items():
        prec = a["tp"] / np.maximum(a["tp"] + a["fp"], 1); rec = a["tp"] / np.maximum(a["tp"] + a["fn"], 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        summ[name] = {"precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist()}
    mean = lambda key, name: np.mean([out["folds"][p][name][key] for p in PONDS], 0).tolist()
    out["alarm"] = summ
    out["base_rate"] = (base / ntot).tolist()
    out["rmse_mean"] = {n: {"all": mean("rmse_all", n), "active": mean("rmse_active", n)} for n in ("Persistence", "Linear AR", "GRU")}
    out["stuck_frac_mean"] = float(np.mean([out["folds"][p]["stuck_frac"] for p in PONDS]))
    (F.OUT / "robustness_alarm.json").write_text(json.dumps(out, indent=2))
    print("stuck fraction (mean over ponds):", round(out["stuck_frac_mean"], 3))
    for n in ("Persistence", "Linear AR", "GRU"):
        print(n, "RMSE all", np.round(out["rmse_mean"][n]["all"], 2), "active", np.round(out["rmse_mean"][n]["active"], 2))
    print("alarm base rate", np.round(out["base_rate"], 3))
    for n, s in summ.items():
        print(n, "P", np.round(s["precision"], 2), "R", np.round(s["recall"], 2), "F1", np.round(s["f1"], 2))


if __name__ == "__main__":
    main()
