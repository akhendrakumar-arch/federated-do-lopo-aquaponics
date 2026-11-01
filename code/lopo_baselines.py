"""Additional baselines under the identical LOPO protocol as lopo_forecast.py.

Linear AR (DO lags only), Ridge (full window), Random forest (summary features),
TCN, CNN-GRU (hybrid in the style of recent DO papers) and a small Transformer.
Same windows, same train-only clipping/scaling, same train/val split, 3 seeds for
neural models.  Results -> results/lopo_baselines.json
"""
import json
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import lopo_forecast as F

warnings.filterwarnings("ignore")
PONDS, HORIZ, DO, SEEDS = F.PONDS, F.HORIZ, F.DO, F.SEEDS
NH = len(HORIZ)


class Wrap(nn.Module):
    """Residual-on-last-DO wrapper, identical output convention to F.Net."""

    def __init__(self, enc, dim):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, NH))

    def forward(self, x):
        return x[:, -1, DO:DO + 1] + self.head(self.enc(x))


class TCN(nn.Module):
    def __init__(self, nf, ch=32):
        super().__init__()
        L, c = [], nf
        for d in (1, 2, 4):
            L += [nn.Conv1d(c, ch, 3, padding=2 * d, dilation=d), nn.ReLU()]
            c = ch
        self.net = nn.Sequential(*L)
        self.k = [2 * d for d in (1, 2, 4)]

    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        return z[:, :, x.shape[1] - 1]  # causal output at the last real time step


class CNNGRU(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(nf, 32, 3, padding=1), nn.ReLU())
        self.gru = nn.GRU(32, 64, batch_first=True)

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.gru(z)
        return o[:, -1]


class TinyTransformer(nn.Module):
    def __init__(self, nf, d=32, win=F.WIN):
        super().__init__()
        self.emb = nn.Linear(nf, d)
        self.pos = nn.Parameter(torch.zeros(1, win, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 64, 0.1, batch_first=True), 2)

    def forward(self, x):
        return self.enc(self.emb(x) + self.pos).mean(1)


BUILD = {"TCN": lambda nf: Wrap(TCN(nf), 32), "CNN-GRU": lambda nf: Wrap(CNNGRU(nf), 64),
         "Transformer": lambda nf: Wrap(TinyTransformer(nf), 32)}


def fit(kind, Xtr, Ytr, Xva, Yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = BUILD[kind](Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=F.LR)
    Xt, Yt, Xv, Yv = map(torch.tensor, (Xtr, Ytr, Xva, Yva))
    best, bad, state = 1e9, 0, None
    for _ in range(F.EPOCHS):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), F.BATCH):
            idx = perm[i:i + F.BATCH]
            loss = nn.functional.smooth_l1_loss(m(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
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


def main():
    t0 = time.time()
    raw = {p: F.windows(F.pond_series(p)) for p in PONDS}
    res = {}
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
        Xte, Yte = raw[held]
        flat = Xtr.reshape(-1, Xtr.shape[2])
        lo, hi = np.percentile(flat, 0.5, 0), np.percentile(flat, 99.5, 0)
        clip = lambda X: np.clip(X, lo, hi)
        mu, sd = clip(flat).mean(0), clip(flat).std(0) + 1e-6
        norm = lambda X: ((clip(X) - mu) / sd).astype(np.float32)
        tgt = lambda Y: ((np.clip(Y, lo[DO], hi[DO]) - mu[DO]) / sd[DO]).astype(np.float32)
        back = lambda z: z * sd[DO] + mu[DO]
        Ntr, Nva, Nte = norm(Xtr), norm(Xva), norm(Xte)
        Ttr, Tva = tgt(Ytr), tgt(Yva)
        fold = {}
        last = lambda N: N[:, -1, DO]
        # linear AR on DO lags only (ARIMA-style linear predictor)
        ar = lambda N: N[:, :, DO]
        m = Ridge(alpha=1.0).fit(ar(Ntr), Ttr - last(Ntr)[:, None])
        fold["Linear AR"] = F.metrics(Yte, back(m.predict(ar(Nte)) + last(Nte)[:, None]))
        # ridge on the whole flattened window
        fl = lambda N: N.reshape(len(N), -1)
        m = Ridge(alpha=10.0).fit(fl(Ntr), Ttr - last(Ntr)[:, None])
        fold["Ridge"] = F.metrics(Yte, back(m.predict(fl(Nte)) + last(Nte)[:, None]))
        # random forest on summary features, multi-output change
        S = [F.summary(a) for a in (Ntr, Nte)]
        m = RandomForestRegressor(150, min_samples_leaf=5, n_jobs=-1, random_state=0)
        m.fit(S[0], Ttr - last(Ntr)[:, None])
        fold["Random forest"] = F.metrics(Yte, back(m.predict(S[1]) + last(Nte)[:, None]))
        for kind in BUILD:
            runs = [F.metrics(Yte, back(F.predict(fit(kind, Ntr, Ttr, Nva, Tva, s), Nte))) for s in SEEDS]
            fold[kind] = {k: np.mean([r[k] for r in runs], 0).tolist() for k in runs[0]}
        res[held] = fold
        print(f"held {held}: " + "  ".join(f"{k}={np.round(v['rmse'], 2).tolist()}" for k, v in fold.items()),
              f"| {time.time() - t0:.0f}s", flush=True)
    agg = {}
    for name in res[PONDS[0]]:
        agg[name] = {}
        for met in ("rmse", "r2"):
            M = np.array([res[p][name][met] for p in PONDS])
            ci = stats.t.ppf(0.975, len(PONDS) - 1) * M.std(0, ddof=1) / np.sqrt(len(PONDS))
            agg[name][met] = {"mean": M.mean(0).tolist(), "ci95": ci.tolist()}
    (F.OUT / "lopo_baselines.json").write_text(json.dumps({"folds": res, "aggregate": agg}, indent=2))
    for k, v in agg.items():
        print(k, "RMSE", np.round(v["rmse"]["mean"], 3), "+/-", np.round(v["rmse"]["ci95"], 3))


if __name__ == "__main__":
    main()
