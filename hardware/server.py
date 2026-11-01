"""Aggregation server (runs on the Raspberry Pi, or on the PC for testing).

Coordinates five leave-one-pond-out folds. In fold k the device whose pond is FOLDS[k] is the
'new farm' (test role); the other four are training clients. All calls block until the other
clients have reported, so devices simply call endpoints in order.

  POST /task         {device, fold}                 -> {role, held} | {done:true}
  POST /r0           {device, fold, hist[201]}      -> {lo, hi}
  POST /r1           {device, fold, n, s1, s2}      -> {mu, sd}
  POST /r2           {device, fold, A[169], B[39]}  -> {W[39]}
  POST /await_model  {device, fold}                 -> {lo, hi, mu, sd, W}   (test device)
  POST /result       {device, fold, n, sse[3], sse_pers[3], ...}
  GET  /status

Run:  python server.py [--port 8000] [--devices esp1:2,esp2:3,esp3:4,esp4:6,esp5:9]
Writes hardware_results.json when all folds are done.
"""
import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hwcommon import FOLDS, NB, percentile_from_hist, solve_ridge

OUT = Path(__file__).resolve().parent / "hardware_results.json"
WAIT = 3600


class Coordinator:
    def __init__(self, devices):
        self.dev = devices                      # device id -> pond
        self.cv = threading.Condition()
        self.cur = 0
        self.rounds, self.agg, self.params, self.results = {}, {}, {}, {}
        self.bin, self.bout = {d: 0 for d in devices}, {d: 0 for d in devices}
        self.t0 = time.time()

    def role(self, dev, fold):
        return "test" if self.dev[dev] == FOLDS[fold] else "train"

    def task(self, dev, fold):
        if fold >= len(FOLDS):
            return {"done": True}
        with self.cv:
            self.cv.wait_for(lambda: self.cur >= fold, WAIT)
        return {"role": self.role(dev, fold), "held": FOLDS[fold]}

    def collect(self, fold, name, dev, payload):
        key = (fold, name)
        n_train = len(self.dev) - 1
        with self.cv:
            self.rounds.setdefault(key, {})[dev] = payload
            self.cv.notify_all()
            if not self.cv.wait_for(lambda: len(self.rounds[key]) >= n_train, WAIT):
                raise TimeoutError(f"fold {fold} round {name}: only {sorted(self.rounds[key])}")
            if key not in self.agg:
                self.agg[key] = self.reduce(fold, name)
                self.cv.notify_all()
            return self.agg[key]

    def reduce(self, fold, name):
        rs = list(self.rounds[(fold, name)].values())
        P = self.params.setdefault(fold, {})
        if name == "r0":
            hist = [sum(r["hist"][i] for r in rs) for i in range(NB + 1)]
            P["lo"], P["hi"] = percentile_from_hist(hist, 0.5), percentile_from_hist(hist, 99.5)
            return {"lo": P["lo"], "hi": P["hi"]}
        if name == "r1":
            n = sum(r["n"] for r in rs)
            mu = sum(r["s1"] for r in rs) / n
            P["mu"], P["sd"] = mu, math.sqrt(max(sum(r["s2"] for r in rs) / n - mu * mu, 0.0)) + 1e-6
            return {"mu": P["mu"], "sd": P["sd"]}
        if name == "r2":
            A = [sum(r["A"][i] for r in rs) for i in range(169)]
            B = [sum(r["B"][i] for r in rs) for i in range(39)]
            P["W"] = solve_ridge(A, B)
            return {"W": P["W"]}
        raise KeyError(name)

    def await_model(self, fold):
        with self.cv:
            self.cv.wait_for(lambda: "W" in self.params.get(fold, {}), WAIT)
            return dict(self.params[fold])

    def result(self, fold, dev, payload):
        with self.cv:
            self.results[fold] = {"device": dev, "held": FOLDS[fold], **payload, **{k: v for k, v in self.params[fold].items()}}
            self.cur = fold + 1
            self.cv.notify_all()
            if self.cur >= len(FOLDS):
                self.finish()

    def finish(self):
        out = {"folds": self.results, "bytes_in": self.bin, "bytes_out": self.bout, "elapsed_s": time.time() - self.t0,
               "devices": self.dev}
        OUT.write_text(json.dumps(out, indent=1))
        print("all folds done -> ", OUT, flush=True)


def make_handler(co):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def _reply(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return len(b)

        def do_GET(self):
            if self.path == "/status":
                self._reply({"cur_fold": co.cur, "devices": co.dev, "elapsed_s": time.time() - co.t0})
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            dev, fold = body.get("device"), body.get("fold", 0)
            try:
                p = self.path
                if p == "/task":
                    r = co.task(dev, fold)
                elif p in ("/r0", "/r1", "/r2"):
                    r = co.collect(fold, p[1:], dev, body)
                elif p == "/await_model":
                    r = co.await_model(fold)
                elif p == "/result":
                    co.result(fold, dev, {k: v for k, v in body.items() if k not in ("device", "fold")})
                    r = {"ok": True}
                else:
                    self.send_error(404)
                    return
                sent = self._reply(r)
                if dev in co.bin:
                    co.bin[dev] += n
                    co.bout[dev] += sent
            except Exception as e:
                self._reply({"error": str(e)}, 500)
    return H


def make_server(devices, port=8000, host="0.0.0.0"):
    co = Coordinator(devices)
    return ThreadingHTTPServer((host, port), make_handler(co)), co


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--devices", default="esp1:2,esp2:3,esp3:4,esp4:6,esp5:9")
    a = ap.parse_args()
    devs = {k: int(v) for k, v in (x.split(":") for x in a.devices.split(","))}
    srv, _ = make_server(devs, a.port)
    print(f"server on :{a.port}, devices {devs}", flush=True)
    srv.serve_forever()
