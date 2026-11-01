# Code and results - Leave-One-Pond-Out Evaluation of Federated DO Forecasting

Code, results and hardware prototype for the manuscript *Leave-One-Pond-Out Evaluation of Federated Dissolved-Oxygen Forecasting on Aquaponics Sensor Data* (submitted to IEEE Internet of Things Journal). Authors: Akhendrakumar Padavala, Md. Kaleem.

- `code/` - audit gate, LOPO classification and forecasting, baselines, conformal recalibration, federated GRU and exact federated linear model, robustness/alarm study (Python; numpy, pandas, scikit-learn, PyTorch).
- `results/` - the JSON/log outputs that the tables and figures in the paper are generated from.
- `hardware/` - Raspberry Pi / ESP32 federated-client prototype, emulator and reference results.
- `run_all.ps1` - re-runs all experiments in order (tens of minutes): `.\run_all.ps1`

## Data (not bundled, public)
Nsukka aquaponics data set (Udanor et al., 2022), Kaggle:
https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets

Place the pond CSV files (IoTPond2.csv, IoTPond3.csv, ...) in `data_raw/csv/` next to `code/`, i.e. `code_and_data/data_raw/csv/`. Pond 1 duplicates Pond 2 and is excluded by the audit gate.

License: MIT (code). The Nsukka data set remains under its own Kaggle terms; the pond `.f32` files in `hardware/data/` are derived from it for the prototype.
