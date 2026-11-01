"""Export each pond's forecasting windows (12 DO inputs + 3 DO targets, float32) for the feeder.

Run on the PC:  python prepare_data.py
Writes data/pond{p}.f32 (n x 15, little-endian float32, chronological) and data/meta.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
sys.path.insert(0, str(HERE))
import lopo_forecast as F  # noqa: E402
from hwcommon import FOLDS, TRAIN_FRAC  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    meta = {}
    for p in FOLDS:
        X, Y = F.windows(F.pond_series(p))
        arr = np.concatenate([X[:, :, F.DO], Y], axis=1).astype("<f4")  # (n, 15)
        arr.tofile(out / f"pond{p}.f32")
        meta[str(p)] = {"n": int(len(arr)), "n_train": int(len(arr) * TRAIN_FRAC)}
        print(f"pond {p}: {len(arr)} windows, {len(arr) * 60 / 1e6:.2f} MB")
    (out / "meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
