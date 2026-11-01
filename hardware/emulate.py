"""Run the whole system on this PC: feeder + server + five emulated devices (threads, real HTTP),
then compare the devices' results with the float64 reference.

    python prepare_data.py && python reference.py && python emulate.py
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "device"))
import feeder  # noqa: E402
import fedclient  # noqa: E402
import server  # noqa: E402
from pc_http import PCHttp  # noqa: E402

DEVS = {"esp1": 2, "esp2": 3, "esp3": 4, "esp4": 6, "esp5": 9}


def main():
    fsrv = feeder.make_server(18001, "127.0.0.1")
    ssrv, co = server.make_server(DEVS, 18000, "127.0.0.1")
    for s in (fsrv, ssrv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    t0 = time.time()
    ths = []
    for d, p in DEVS.items():
        cfg = {"device": d, "pond": p, "server": "http://127.0.0.1:18000", "feeder": "http://127.0.0.1:18001"}
        th = threading.Thread(target=fedclient.run, args=(cfg, PCHttp(), lambda m, d=d: print(f"[{d}] {m}", flush=True)))
        th.start(); ths.append(th)
    for th in ths:
        th.join()
    print(f"emulation finished in {time.time() - t0:.0f}s")
    res = json.loads((HERE / "hardware_results.json").read_text())
    ref = json.loads((HERE / "reference_results.json").read_text())
    print("\npond | RMSE device (15m,1h,3h) | RMSE reference | max rel diff | bytes up/down (device)")
    worst = 0.0
    for k, r in res["folds"].items():
        h = str(r["held"])
        rm = [(s / r["n"]) ** 0.5 for s in r["sse"]]
        rd = max(abs(a - b) / b for a, b in zip(rm, ref[h]["rmse"]))
        dd = max(abs(a - b) for a, b in zip(r["W"], ref[h]["W"]))
        worst = max(worst, rd)
        dev = r["device"]
        print(h, np.round(rm, 4), np.round(ref[h]["rmse"], 4), f"{rd:.2e}", f"|W diff {dd:.1e}", res["bytes_in"][dev], res["bytes_out"][dev])
    print(f"worst relative RMSE difference device vs reference: {worst:.2e}")


if __name__ == "__main__":
    main()
