"""Portable federated linear-AR client. Runs unchanged on an ESP32 (MicroPython) and on CPython
(for emulation). Pure Python, no numpy.

`http` must provide:  get_json(url) -> dict,  get_bytes(url) -> bytes,  post_json(url, obj) -> dict.
cfg keys: device, pond, server (http://host:8000), feeder (http://host:8001).
"""
import json
import time
from array import array

try:
    import gc
    _mem = gc.mem_free
except Exception:  # CPython
    def _mem():
        return 0

try:
    from time import ticks_diff, ticks_ms
except ImportError:
    def ticks_ms():
        return int(time.time() * 1000)

    def ticks_diff(a, b):
        return a - b

WIN, NH, NF = 12, 3, 13
HMAX, NB = 60.0, 200
BW = HMAX / NB
CH = 100  # windows per chunk (6 kB) - small enough for the ESP32 heap


def _chunks(http, feed, pond, total):
    s = 0
    while s < total:
        m = min(CH, total - s)
        a = array("f")
        a.frombytes(http.get_bytes("%s/data?pond=%d&start=%d&count=%d" % (feed, pond, s, m)))
        yield a, m
        s += m


def _train(http, cfg, k, log):
    dev, pond, srv, feed = cfg["device"], cfg["pond"], cfg["server"], cfg["feeder"]
    meta = http.get_json("%s/meta?pond=%d" % (feed, pond))
    nt = meta["n_train"]
    t0 = ticks_ms()
    # R0: histogram of DO input values -> clip bounds
    hist = [0] * (NB + 1)
    for a, m in _chunks(http, feed, pond, nt):
        for j in range(m):
            b = j * 15
            for i in range(WIN):
                v = a[b + i]
                idx = 0 if v < 0 else int(v / BW)
                hist[idx if idx < NB else NB] += 1
    r = http.post_json(srv + "/r0", {"device": dev, "fold": k, "hist": hist})
    lo, hi = r["lo"], r["hi"]
    # R1: sums of clipped values -> mean / std
    n = 0
    s1 = 0.0
    s2 = 0.0
    for a, m in _chunks(http, feed, pond, nt):
        for j in range(m):
            b = j * 15
            for i in range(WIN):
                v = a[b + i]
                v = lo if v < lo else (hi if v > hi else v)
                s1 += v
                s2 += v * v
        n += m * WIN
    r = http.post_json(srv + "/r1", {"device": dev, "fold": k, "n": n, "s1": s1, "s2": s2})
    mu, sd = r["mu"], r["sd"]
    # R2: sufficient statistics A = sum d d^T, B = sum d t^T
    A = [0.0] * (NF * NF)
    B = [0.0] * (NF * NH)
    for a, m in _chunks(http, feed, pond, nt):
        for j in range(m):
            b = j * 15
            d = [0.0] * NF
            for i in range(WIN):
                v = a[b + i]
                d[i] = ((lo if v < lo else (hi if v > hi else v)) - mu) / sd
            d[WIN] = 1.0
            last = d[WIN - 1]
            t = [0.0] * NH
            for h in range(NH):
                v = a[b + WIN + h]
                t[h] = ((lo if v < lo else (hi if v > hi else v)) - mu) / sd - last
            for i in range(NF):
                di = d[i]
                row = i * NF
                for q in range(NF):
                    A[row + q] += di * d[q]
                row = i * NH
                for h in range(NH):
                    B[row + h] += di * t[h]
    t_stat = ticks_diff(ticks_ms(), t0)
    r = http.post_json(srv + "/r2", {"device": dev, "fold": k, "A": A, "B": B, "t_ms": t_stat, "mem_free": _mem(), "n_windows": nt})
    log("fold %d train done: %d windows, %d ms, W[0]=%.4f" % (k, nt, t_stat, r["W"][0]))


def _test(http, cfg, k, log):
    dev, pond, srv, feed = cfg["device"], cfg["pond"], cfg["server"], cfg["feeder"]
    meta = http.get_json("%s/meta?pond=%d" % (feed, pond))
    nt = meta["n"]
    p = http.post_json(srv + "/await_model", {"device": dev, "fold": k})
    lo, hi, mu, sd, W = p["lo"], p["hi"], p["mu"], p["sd"], p["W"]
    t0 = ticks_ms()
    sse = [0.0] * NH
    sse_p = [0.0] * NH
    ssum = [0.0] * NH
    first = []
    cnt = 0
    for a, m in _chunks(http, feed, pond, nt):
        for j in range(m):
            b = j * 15
            d = [0.0] * NF
            for i in range(WIN):
                v = a[b + i]
                d[i] = ((lo if v < lo else (hi if v > hi else v)) - mu) / sd
            d[WIN] = 1.0
            last = d[WIN - 1]
            lastc = last * sd + mu
            for h in range(NH):
                z = last
                for i in range(NF):
                    z += d[i] * W[i * NH + h]
                pred = z * sd + mu
                y = a[b + WIN + h]
                sse[h] += (y - pred) * (y - pred)
                sse_p[h] += (y - lastc) * (y - lastc)
                ssum[h] += pred
                if cnt < 5:
                    first.append(pred)
            cnt += 1
    ms = ticks_diff(ticks_ms(), t0)
    http.post_json(srv + "/result", {"device": dev, "fold": k, "n": cnt, "sse": sse, "sse_pers": sse_p, "sum_pred": ssum,
                                     "first": first, "t_ms": ms, "ms_per_window": ms / max(cnt, 1), "mem_free": _mem()})
    log("fold %d test done: %d windows, %d ms, RMSE %s" % (k, cnt, ms, [round((s / cnt) ** 0.5, 3) for s in sse]))


def run(cfg, http, log=print):
    k = 0
    while True:
        t = http.post_json(cfg["server"] + "/task", {"device": cfg["device"], "fold": k})
        if t.get("done"):
            log("all folds finished")
            return
        (_test if t["role"] == "test" else _train)(http, cfg, k, log)
        k += 1
