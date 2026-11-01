"""HTTP shim for CPython (emulation and PC testing). Same interface as esp_http.py."""
import json
import urllib.request


class PCHttp:
    def get_bytes(self, url):
        with urllib.request.urlopen(url, timeout=3600) as r:
            return r.read()

    def get_json(self, url):
        return json.loads(self.get_bytes(url))

    def post_json(self, url, obj):
        req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read())
        if "error" in out:
            raise RuntimeError(out["error"])
        return out
