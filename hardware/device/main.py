"""ESP32 entry point: connect to Wi-Fi, then run the federated linear-AR client."""
import gc
import time

import network

import config
import fedclient
from esp_http import EspHttp


def wifi():
    sta = network.WLAN(network.STA_IF)
    sta.active(True)
    if not sta.isconnected():
        sta.connect(config.WIFI_SSID, config.WIFI_PASSWORD)
        t = time.ticks_ms()
        while not sta.isconnected():
            if time.ticks_diff(time.ticks_ms(), t) > 30000:
                raise RuntimeError("Wi-Fi connect timeout")
            time.sleep_ms(200)
    print("Wi-Fi ok, IP", sta.ifconfig()[0])


wifi()
gc.collect()
cfg = {"device": config.DEVICE, "pond": config.POND, "server": config.SERVER, "feeder": config.FEEDER}
fedclient.run(cfg, EspHttp(), log=lambda m: print("[%s] %s" % (config.DEVICE, m)))
