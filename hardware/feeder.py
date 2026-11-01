"""Test-data feeder (runs on the PC). Serves each pond's recorded windows to its ESP32 over HTTP.

  GET /meta?pond=2                       -> {"n": 15177, "n_train": 12900}
  GET /data?pond=2&start=0&count=150     -> raw little-endian float32, count x 15 values

The feeder stands in for the pond's sensor: the device 'measures' by pulling recorded windows.
Run:  python feeder.py [port]      (default 8001)
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DATA = Path(__file__).resolve().parent / "data"
ROW = 15 * 4  # bytes per window
_META = None
_BLOB = {}
_LOCK = threading.Lock()


def _meta():
    global _META
    if _META is None:
        _META = json.loads((DATA / "meta.json").read_text())
    return _META


def _blob(p):
    with _LOCK:
        if p not in _BLOB:
            _BLOB[p] = (DATA / f"pond{p}.f32").read_bytes()
        return _BLOB[p]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/meta":
                self._send(json.dumps(_meta()[q["pond"]]).encode(), "application/json")
            elif u.path == "/data":
                b = _blob(q["pond"])
                s, c = int(q["start"]), int(q["count"])
                self._send(b[s * ROW:(s + c) * ROW], "application/octet-stream")
            else:
                self.send_error(404)
        except Exception as e:  # keep serving other devices
            self.send_error(500, str(e))


def make_server(port=8001, host="0.0.0.0"):
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print(f"feeder on :{port}, data from {DATA}")
    make_server(port).serve_forever()
