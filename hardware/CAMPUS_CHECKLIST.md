# Campus session checklist (hardware demo)

## Bring / prepare BEFORE going (do at home, needs internet)
- [ ] Laptop with this folder (`journal_paper\hardware`) and `pip install esptool mpremote` already done.
- [ ] MicroPython `.bin` for your ESP32 model downloaded (micropython.org/download); note the chip type (classic ESP32 vs S3/C3 -> flash offset 0x1000 vs 0x0).
- [ ] USB driver for the boards' chip installed (CP210x or CH340) and a USB data cable for each board (some cables are charge-only).
- [ ] 5 ESP32 boards, 1 Raspberry Pi 3/4/5, its power supply, microSD card with Raspberry Pi OS (or Raspberry Pi Imager ready), USB hub if you want all five at once.
- [ ] A PRIVATE network the boards can share: a phone hotspot or a small travel router is safest. Campus Wi-Fi often isolates devices
      (clients cannot reach each other) or needs a login page that ESP32s cannot pass.
- [ ] Re-run once at home to be sure nothing broke: `python emulate.py` (should print worst relative difference ~1e-14).

## On campus, in order
1. Make the network: hotspot/router on; note the Wi-Fi name and password. Connect laptop and Pi to it.
2. Pi: find its IP (router list or `ping raspberrypi.local`). Copy `server.py` + `hwcommon.py` (`scp`), run `python3 server.py`.
3. Laptop: `ipconfig` -> note the Wi-Fi IPv4 address. Run `python feeder.py`; allow port 8001 in the Windows firewall (private network).
4. Plug in the five boards; read their COM ports (Device Manager > Ports).
5. Flash + install: `.\flash_all.ps1 -Ports COM..,.. -Firmware <bin> -Ssid .. -Password .. -ServerIp <Pi IP> -FeederIp <laptop IP>`
6. Press RESET on each board. Watch serial output (`python -m mpremote connect COMx repl`). Pi prints `all folds done`.
7. Save `hardware_results.json` from the Pi, plus any serial errors/screenshots. Send them for comparison with `reference_results.json`.

## Record for the paper / proposal
- Photo of the setup (5 boards + Pi + laptop), router/hotspot model, board model, MicroPython version.
- From `hardware_results.json`: per-pond RMSE, `t_ms`, `ms_per_window`, `mem_free`, bytes in/out per device.
- Anything that failed and how it was fixed (honest log).

## Not claimed
Live pond deployment. Data are recorded windows replayed from the laptop.
