# Flash MicroPython onto five ESP32 boards from this laptop and install the federated client.
#
# Prerequisites (once):   pip install esptool mpremote
# Download the MicroPython .bin for your exact board from https://micropython.org/download/ (ESP32_GENERIC for a plain ESP32).
# Plug ALL five boards in (USB data cables / hub); find their COM ports in Device Manager > Ports.
#
# Example:
#   .\flash_all.ps1 -Ports COM5,COM6,COM7,COM8,COM9 -Firmware C:\fw\ESP32_GENERIC.bin `
#       -Ssid "MyWiFi" -Password "secret" -ServerIp 192.168.1.50 -FeederIp 192.168.1.20
#
# -FlashOffset: 0x1000 for classic ESP32 (default), 0x0 for ESP32-S3 / C3 (check the page for your board).
# If esptool stalls on "Connecting...", hold the board's BOOT button until it starts.
# Board i gets DEVICE=esp<i> and the pond below: esp1:2 esp2:3 esp3:4 esp4:6 esp5:9 (must match server.py --devices).
param(
  [Parameter(Mandatory)][string[]]$Ports,
  [Parameter(Mandatory)][string]$Firmware,
  [Parameter(Mandatory)][string]$Ssid,
  [Parameter(Mandatory)][string]$Password,
  [Parameter(Mandatory)][string]$ServerIp,
  [Parameter(Mandatory)][string]$FeederIp,
  [string]$FlashOffset = "0x1000",
  [switch]$SkipFlash
)
$ErrorActionPreference = "Stop"
$ponds = 2, 3, 4, 6, 9
if ($Ports.Count -ne 5) { throw "Give exactly five COM ports (got $($Ports.Count))." }
if (-not (Test-Path $Firmware)) { throw "Firmware file not found: $Firmware" }
$dev = Join-Path $PSScriptRoot "device"
$tmp = Join-Path $env:TEMP "fed_config"
New-Item -ItemType Directory -Force $tmp | Out-Null

for ($i = 0; $i -lt 5; $i++) {
  $port = $Ports[$i]; $name = "esp$($i + 1)"; $pond = $ponds[$i]
  Write-Host "=== $name (pond $pond) on $port ===" -ForegroundColor Cyan
  $cfg = Join-Path $tmp "config.py"
  @"
WIFI_SSID = "$Ssid"
WIFI_PASSWORD = "$Password"
DEVICE = "$name"
POND = $pond
SERVER = "http://${ServerIp}:8000"
FEEDER = "http://${FeederIp}:8001"
"@ | Set-Content $cfg -Encoding ASCII
  if (-not $SkipFlash) {
    python -m esptool --port $port erase_flash
    python -m esptool --port $port --baud 460800 write_flash -z $FlashOffset $Firmware
    Start-Sleep -Seconds 5   # board reboots into MicroPython
  }
  foreach ($f in "fedclient.py", "esp_http.py", "main.py") {
    python -m mpremote connect $port fs cp (Join-Path $dev $f) ":$f"
  }
  python -m mpremote connect $port fs cp $cfg ":config.py"
  Write-Host "$name done." -ForegroundColor Green
}
Remove-Item $cfg -Force
Write-Host "`nAll five boards programmed. Next: start server.py on the Pi, feeder.py on this laptop, then press RESET on each board." -ForegroundColor Yellow
