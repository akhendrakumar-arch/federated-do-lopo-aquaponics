# Federated linear-AR demo: 5 ESP32 + 1 Raspberry Pi (replayed pond data)

**What it shows.** Five edge devices, one per pond, train a shared dissolved-oxygen forecaster
without sending raw readings anywhere. Each device sends only small summary tables to a Raspberry Pi,
which solves for the 39 model coefficients and sends them back. The leave-one-pond-out experiment of
the paper is repeated on hardware (5 rounds; each device is the "new farm" once).
**What it does not show.** A live pond deployment: the PC *replays recorded* windows to each device.

```
PC  (feeder.py: replays each pond's recorded data)
 | Wi-Fi (HTTP GET)
ESP32 x5  (device/fedclient.py: computes summaries on-device)
 | Wi-Fi (HTTP POST, small JSON)
Raspberry Pi  (server.py: combines summaries, solves, returns coefficients)
```

## Already verified on the PC (no hardware needed)
`python prepare_data.py && python reference.py && python emulate.py` runs the feeder, the server and five
emulated devices (the same `fedclient.py` that goes on the ESP32) over real HTTP. Device results equal the float64
reference to ~1e-14, and the reference reproduces the paper's pooled Linear AR RMSE to 3 decimals
(Pond 9: within 0.002). **The firmware itself has not been run on a real ESP32 yet.**

## Files
| file | runs on | purpose |
|---|---|---|
| `prepare_data.py` | PC | exports each pond's windows to `data/` (done; ~3.5 MB) |
| `feeder.py` | PC | serves windows to devices (`python feeder.py`, port 8001) |
| `server.py` | Raspberry Pi | aggregator (`python3 server.py`, port 8000, needs only Python 3) |
| `reference.py` | PC | float64 reference results to check the hardware against |
| `device/fedclient.py`, `esp_http.py`, `main.py`, `config_example.py` | each ESP32 | MicroPython firmware |
| `emulate.py` | PC | whole system on one machine |

## Setup (about half a day)
1. **ESP32 boards.** Flash MicroPython (ESP32 generic build from micropython.org). Use Thonny or `mpremote`.
2. **Copy to each board:** `device/fedclient.py`, `device/esp_http.py`, `device/main.py`, and a `config.py` made from
   `config_example.py` (set `DEVICE` and `POND`: esp1:2, esp2:3, esp3:4, esp4:6, esp5:9; Wi-Fi name/password;
   the Pi's and PC's IP addresses). All devices must be on the same Wi-Fi network as the Pi and the PC.
3. **Raspberry Pi.** Copy `server.py` and `hwcommon.py`; run `python3 server.py`. (Optional: `--devices esp1:2,...` to rename.)
4. **PC.** In this folder run `python feeder.py`. Allow port 8001 in the Windows firewall (private network).
5. **Start the boards** (reset them; `main.py` runs automatically). Watch the serial output of each. The Pi prints
   `all folds done` and writes `hardware_results.json`.
6. **Check.** Compare `hardware_results.json` with `reference_results.json` (per-pond RMSE, coefficients `W`).
   Run `python emulate.py`'s comparison logic on the Pi's file, or send me both files.

## What to expect and record
- **Accuracy.** MicroPython on the ESP32 uses single-precision floats, so RMSE should agree with the reference to
  roughly 3-4 significant digits (not 1e-14). A larger gap means a bug or lost packets: tell me.
- **Time.** Training pass (about 12k windows) took ~3 s per fold in the CPython emulation; expect tens of seconds on an
  ESP32. `t_ms` per device and `ms_per_window` are logged to the server automatically.
- **Memory.** `mem_free` (bytes of heap left) is logged. If a board runs out of memory, lower `CH` in `fedclient.py` (default 100 windows).
- **Network.** The server logs bytes in/out per device (`bytes_in`, `bytes_out`). Transport is JSON text, so it is larger than
  the compact payload: the sufficient statistics themselves are 13x13 + 13x3 values = 832 bytes in 32-bit floats.
  Measured in emulation: ~21 kB up and ~5 kB down per device over all five folds, including histograms and clipping statistics.

## Known risks (untested on hardware)
- `urequests` has no timeout; the server holds requests until all clients have reported (by design). If one board dies, the others wait.
  Restart everything (server first).
- Very old MicroPython builds may lack `array.frombytes` or `time.ticks_diff`; use a release from 2023 or later.
- A different Wi-Fi band or a router that isolates clients (guest networks) will block device-to-Pi traffic.
