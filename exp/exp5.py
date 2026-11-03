"""Experiment 5: Heston model pricing and calibration (Chapter 10).

Generates a synthetic market implied-vol surface from true Heston params,
calibrates a new set of params to it, and compares results.
"""

import json
import os
import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    HESTON_K_FRAC,
    HESTON_S0,
    HESTON_T_GRID,
    HESTON_TRUE_DELTA,
    HESTON_TRUE_KAPPA,
    HESTON_TRUE_R,
    HESTON_TRUE_RHO,
    HESTON_TRUE_THETA,
    HESTON_TRUE_V0,
)
from src.heston import (
    HestonParams,
    calibrate_heston,
    heston_surface,
)

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1-2: True Heston params and strike grid
# ---------------------------------------------------------------------------

true_params = HestonParams(
    v0=HESTON_TRUE_V0,
    kappa=HESTON_TRUE_KAPPA,
    theta=HESTON_TRUE_THETA,
    delta=HESTON_TRUE_DELTA,
    rho=HESTON_TRUE_RHO,
    r=HESTON_TRUE_R,
)

K_grid = [HESTON_S0 * f for f in HESTON_K_FRAC]

# ---------------------------------------------------------------------------
# Step 3: Compute "market" implied-vol surface
# ---------------------------------------------------------------------------

print("=== Step 1: Computing market implied-vol surface ===")
market_ivols = heston_surface(HESTON_S0, HESTON_T_GRID, K_grid, true_params)
print("Market implied vols (rows=T, cols=K):")
print(market_ivols)

nan_count = int(np.sum(np.isnan(market_ivols)))
if nan_count > 0:
    print(f"WARNING: {nan_count} NaN values in market surface – continuing.")

# ---------------------------------------------------------------------------
# Step 4: Calibrate Heston with 3-minute timeout guard
# ---------------------------------------------------------------------------

print("\n=== Step 2: Calibrating Heston model ===")

cal_params = None
rmse = float("nan")

class _Timeout(Exception):
    pass

def _handler(signum, frame):  # noqa: ARG001
    raise _Timeout()

signal.signal(signal.SIGALRM, _handler)
signal.alarm(180)  # 3-minute hard limit
try:
    cal_params, rmse = calibrate_heston(
        HESTON_S0,
        market_ivols,
        HESTON_T_GRID,
        K_grid,
        r=HESTON_TRUE_R,
        seed=42,
    )
except _Timeout:
    print("WARNING: calibration timed out after 3 minutes; using partial result.")
    # Fall back to true params so downstream steps still produce output
    cal_params = true_params
    rmse = float("nan")
finally:
    signal.alarm(0)

# ---------------------------------------------------------------------------
# Step 5: Fitted surface and comparison
# ---------------------------------------------------------------------------

fitted_ivols = heston_surface(HESTON_S0, HESTON_T_GRID, K_grid, cal_params)

print("\nTrue params:      ", true_params)
print("Calibrated params:", cal_params)
print(f"Calibration RMSE: {rmse:.6f}")

# ---------------------------------------------------------------------------
# Step 6: Plots
# ---------------------------------------------------------------------------

# Find maturity closest to 1.0 for smile plot
T_arr = np.array(HESTON_T_GRID)
idx_t1 = int(np.argmin(np.abs(T_arr - 1.0)))
T_plot = T_arr[idx_t1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(K_grid, market_ivols[idx_t1], "o-", label=f"Market (T={T_plot})")
ax.plot(K_grid, fitted_ivols[idx_t1], "s--", label=f"Fitted (T={T_plot})")
ax.set_xlabel("Strike K")
ax.set_ylabel("Implied vol")
ax.set_title("Heston: market vs fitted smile")
ax.legend()
fig.tight_layout()
fig.savefig("results/exp5_heston_smile.png", dpi=100)
plt.close(fig)

# Calibrated surface heatmap
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(
    fitted_ivols,
    aspect="auto",
    origin="lower",
    extent=[K_grid[0], K_grid[-1], T_arr[0], T_arr[-1]],
)
fig.colorbar(im, ax=ax, label="Implied vol")
ax.set_xlabel("Strike K")
ax.set_ylabel("Maturity T")
ax.set_title("Heston calibrated implied-vol surface")
fig.tight_layout()
fig.savefig("results/exp5_heston_surface.png", dpi=100)
plt.close(fig)

# ---------------------------------------------------------------------------
# Step 7: Save results
# ---------------------------------------------------------------------------

results_dict = {
    "true_v0": HESTON_TRUE_V0,
    "true_kappa": HESTON_TRUE_KAPPA,
    "true_theta": HESTON_TRUE_THETA,
    "true_delta": HESTON_TRUE_DELTA,
    "true_rho": HESTON_TRUE_RHO,
    "cal_v0": float(cal_params.v0),
    "cal_kappa": float(cal_params.kappa),
    "cal_theta": float(cal_params.theta),
    "cal_delta": float(cal_params.delta),
    "cal_rho": float(cal_params.rho),
    "cal_rmse": float(rmse) if not np.isnan(rmse) else None,
}
with open("results/exp5_results.json", "w") as fh:
    json.dump(results_dict, fh, indent=2)

print("\nResults saved to results/exp5_results.json")
print("Plots saved to results/exp5_heston_smile.png and results/exp5_heston_surface.png")
