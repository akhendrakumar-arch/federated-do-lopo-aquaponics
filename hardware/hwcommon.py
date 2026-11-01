"""Shared constants and tiny pure-Python numerics for the federated linear-AR hardware demo.

Used by the aggregation server (Raspberry Pi / PC) and the numpy reference.  The ESP32
firmware re-implements only the per-window arithmetic (device/fedclient.py).

Protocol (leave-one-pond-out fold with held-out pond h; clients = the other four ponds):
  R0  each client sends a histogram of its DO input values        -> server: clip bounds lo/hi
  R1  each client sends n, sum, sum-of-squares of CLIPPED values  -> server: mu, sd
  R2  each client sends A = sum d d^T (13x13), B = sum d t^T (13x3) -> server solves ridge -> W (13x3)
  The held-out device receives lo, hi, mu, sd, W and forecasts its own windows.
A window is 12 DO inputs + 3 DO targets (15 float32 values); d = [normalised 12 inputs, 1];
t_h = normalised clipped target_h - normalised last input (the network predicts the change).
"""
WIN, NH, NF = 12, 3, 13          # window length, horizons, features incl. bias
HMAX, NB = 60.0, 200             # histogram range [0, 60), 200 bins + one overflow bin
BW = HMAX / NB
ALPHA = 1.0                      # ridge penalty (bias not penalised)
FOLDS = [2, 3, 4, 6, 9]          # held-out pond per fold
TRAIN_FRAC = 0.85                # chronological training share of each client's windows


def bin_index(v):
    if v < 0:
        return 0
    i = int(v / BW)
    return i if i < NB else NB


def percentile_from_hist(counts, q):
    """q in [0, 100]; linear interpolation inside the bin."""
    total = float(sum(counts))
    target = q / 100.0 * total
    cum = 0.0
    for i, c in enumerate(counts):
        if c > 0 and cum + c >= target:
            return (i + (target - cum) / c) * BW
        cum += c
    return HMAX


def solve_ridge(A, B, alpha=ALPHA):
    """Solve (A + alpha*I0) W = B by Gaussian elimination. A is NFxNF row-major list, B is NFxNH row-major."""
    n, m = NF, NH
    M = [[A[i * n + j] + (alpha if (i == j and i < n - 1) else 0.0) for j in range(n)] + [B[i * m + h] for h in range(m)]
         for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for j in range(c, n + m):
            M[c][j] /= piv
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                for j in range(c, n + m):
                    M[r][j] -= f * M[c][j]
    return [M[i][n + h] for i in range(n) for h in range(m)]
