"""Federated dissolved-oxygen forecasting with ponds as clients (leave-one-pond-out).

Each training pond is a client that never shares raw data; only model weights (and
scalar normalisation statistics) are exchanged. The held-out pond plays a brand-new
farm that joins after training. All methods are scored on the same windows of the
held-out pond: everything after its first 15% (which is used only for few-shot
personalisation) and after a gap that prevents window overlap.

Methods: Local-only (mean over single-pond models), Centralised (pooled data, upper
reference), FedAvg, FedProx, and FedAvg + few-shot personalisation.
"""
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

import lopo_forecast as F

PONDS, HORIZ, DO = F.PONDS, F.HORIZ, F.DO
ROUNDS, LOCAL_EPOCHS, BATCH, LR = 20, 1, 256, 2e-3
MU_PROX = 0.01
CAP = 12000          # max training windows per client
SEEDS = [0, 1]
ADAPT_FRAC, GAP = 0.15, 48


def local_train(model, X, Y, epochs, lr=LR, mu=0.0, ref=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = nn.functional.smooth_l1_loss(model(X[idx]), Y[idx])
            if mu > 0:
                loss = loss + 0.5 * mu * sum(((p - r) ** 2).sum() for p, r in zip(model.parameters(), ref))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def val_loss(model, X, Y):
    model.eval()
    with torch.no_grad():
        return nn.functional.mse_loss(model(X), Y).item()


def fedavg(clients, vals, seed, mu=0.0):
    """clients/vals: lists of (X, Y) tensors. Returns best global model by weighted val loss."""
    torch.manual_seed(seed)
    g = F.Net("GRU", clients[0][0].shape[2])
    n = np.array([len(c[0]) for c in clients], float)
    w = n / n.sum()
    best, state = 1e9, None
    for r in range(ROUNDS):
        ref = [p.detach().clone() for p in g.parameters()]
        locals_ = []
        for X, Y in clients:
            m = copy.deepcopy(g)
            local_train(m, X, Y, LOCAL_EPOCHS, mu=mu, ref=ref)
            locals_.append(m.state_dict())
        new = {k: sum(wi * s[k] for wi, s in zip(w, locals_)) for k in locals_[0]}
        g.load_state_dict(new)
        v = sum(wi * val_loss(g, *vals[i]) for i, wi in enumerate(w))
        if v < best:
            best, state = v, {k: x.clone() for k, x in g.state_dict().items()}
    g.load_state_dict(state)
    return g, ROUNDS


def main():
    t0 = time.time()
    raw = {p: F.windows(F.pond_series(p)) for p in PONDS}
    res = {}
    n_params = None
    for held in PONDS:
        train_ps = [p for p in PONDS if p != held]
        tr, va = {}, {}
        for p in train_ps:
            X, Y = raw[p]
            k = int(len(X) * 0.85)
            tr[p], va[p] = (X[:k], Y[:k]), (X[k:], Y[k:])
        # federated normalisation statistics: each client reports sums / percentiles only
        stat = {}
        for p in train_ps:
            fl = tr[p][0].reshape(-1, tr[p][0].shape[2])
            stat[p] = (len(fl), fl.sum(0), (fl ** 2).sum(0), np.percentile(fl, 0.5, 0), np.percentile(fl, 99.5, 0))
        N = sum(s[0] for s in stat.values())
        mu = sum(s[1] for s in stat.values()) / N
        sd = np.sqrt(sum(s[2] for s in stat.values()) / N - mu ** 2) + 1e-6
        lo = np.mean([s[3] for s in stat.values()], 0); hi = np.mean([s[4] for s in stat.values()], 0)
        norm = lambda X: ((np.clip(X, lo, hi) - mu) / sd).astype(np.float32)
        tgt = lambda Y: ((np.clip(Y, lo[DO], hi[DO]) - mu[DO]) / sd[DO]).astype(np.float32)
        back = lambda z: z * sd[DO] + mu[DO]
        rng = np.random.default_rng(0)

        def pack(X, Y, cap=None):
            if cap and len(X) > cap:
                s = rng.choice(len(X), cap, replace=False); X, Y = X[s], Y[s]
            return torch.tensor(norm(X)), torch.tensor(tgt(Y))

        clients = [pack(*tr[p], CAP) for p in train_ps]
        vals = [pack(*va[p]) for p in train_ps]
        Xh, Yh = raw[held]
        a = int(len(Xh) * ADAPT_FRAC)
        Xad, Yad = pack(Xh[:a], Yh[:a])
        Xte, Yte = Xh[a + GAP:], Yh[a + GAP:]
        Nte = torch.tensor(norm(Xte))
        score = lambda m: F.metrics(Yte, back(F.predict(m, Nte.numpy())))
        out = {"Persistence": F.metrics(Yte, np.repeat(np.clip(Xte, lo, hi)[:, -1, DO:DO + 1], len(HORIZ), 1)),
               "n_test": len(Xte)}
        acc = {k: [] for k in ("Local", "Centralised", "FedAvg", "FedProx", "FedAvg+FT")}
        for s in SEEDS:
            # Local-only: every client trains alone, score each on the new pond, average
            loc = []
            for (X, Y), (Xv, Yv) in zip(clients, vals):
                torch.manual_seed(s)
                m = F.Net("GRU", X.shape[2])
                local_train(m, X, Y, ROUNDS)
                loc.append(score(m))
            acc["Local"].append({k: np.mean([l[k] for l in loc], 0).tolist() for k in loc[0]})
            # Centralised on pooled data (privacy-violating upper reference)
            Xc = torch.cat([c[0] for c in clients]); Yc = torch.cat([c[1] for c in clients])
            Xv = torch.cat([v[0] for v in vals]); Yv = torch.cat([v[1] for v in vals])
            cm = F.fit_nn("GRU", Xc.numpy(), Yc.numpy(), Xv.numpy(), Yv.numpy(), s)
            acc["Centralised"].append(score(cm))
            g, rounds = fedavg(clients, vals, s)
            acc["FedAvg"].append(score(g))
            n_params = sum(p.numel() for p in g.parameters())
            gp, _ = fedavg(clients, vals, s, mu=MU_PROX)
            acc["FedProx"].append(score(gp))
            ft = copy.deepcopy(g)
            local_train(ft, Xad, Yad, 3, lr=5e-4)
            acc["FedAvg+FT"].append(score(ft))
        for k, v in acc.items():
            out[k] = {m: np.mean([x[m] for x in v], 0).tolist() for m in v[0]}
        res[held] = out
        print(f"held {held}: n={len(Xte)} RMSE " + "  ".join(
            f"{k}={np.round(out[k]['rmse'], 2).tolist()}" for k in
            ("Persistence", "Local", "Centralised", "FedAvg", "FedProx", "FedAvg+FT")), f"| {time.time() - t0:.0f}s", flush=True)

    agg = {}
    for k in ("Persistence", "Local", "Centralised", "FedAvg", "FedProx", "FedAvg+FT"):
        M = np.array([res[p][k]["rmse"] for p in PONDS])
        ci = stats.t.ppf(0.975, len(PONDS) - 1) * M.std(0, ddof=1) / np.sqrt(len(PONDS))
        agg[k] = {"rmse_mean": M.mean(0).tolist(), "rmse_ci95": ci.tolist(),
                  "r2_mean": np.mean([res[p][k]["r2"] for p in PONDS], 0).tolist()}
    comm_mb = 2 * n_params * 4 * (len(PONDS) - 1) * ROUNDS / 1e6
    out = {"horizons_min": [h * 5 for h in HORIZ], "folds": res, "aggregate": agg, "gru_params": n_params,
           "rounds": ROUNDS, "clients": len(PONDS) - 1, "comm_total_MB_per_training": comm_mb,
           "comm_per_round_per_client_KB": 2 * n_params * 4 / 1e3, "mu_prox": MU_PROX, "seeds": SEEDS,
           "note": "all methods scored on held-out pond excluding first 15% (+gap) used for personalisation"}
    (F.OUT / "fed_forecast.json").write_text(json.dumps(out, indent=2))
    for k, v in agg.items():
        print(k, "RMSE", np.round(v["rmse_mean"], 3), "+/-", np.round(v["rmse_ci95"], 3))
    print(f"communication: {comm_mb:.1f} MB total, {out['comm_per_round_per_client_KB']:.1f} KB per client-round")


if __name__ == "__main__":
    main()
