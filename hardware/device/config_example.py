# Copy to config.py on each ESP32 and edit. One ESP32 = one pond.
WIFI_SSID = "YOUR_WIFI_NAME"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"

DEVICE = "esp1"                       # esp1..esp5, must match the server's --devices list
POND = 2                              # esp1:2  esp2:3  esp3:4  esp4:6  esp5:9
SERVER = "http://192.168.1.50:8000"   # Raspberry Pi address (python3 server.py)
FEEDER = "http://192.168.1.20:8001"   # PC address (python feeder.py)
