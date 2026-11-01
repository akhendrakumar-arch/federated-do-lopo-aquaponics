"""HTTP helper for ESP32 MicroPython (urequests / requests). Same interface as pc_http.py."""
import json

try:
    import urequests as requests
except ImportError:
    import requests


class EspHttp:
    def get_bytes(self, url):
        r = requests.get(url)
        try:
            return r.content
        finally:
            r.close()

    def get_json(self, url):
        r = requests.get(url)
        try:
            return r.json()
        finally:
            r.close()

    def post_json(self, url, obj):
        r = requests.post(url, data=json.dumps(obj), headers={"Content-Type": "application/json"})
        try:
            out = r.json()
        finally:
            r.close()
        if "error" in out:
            raise RuntimeError(out["error"])
        return out
