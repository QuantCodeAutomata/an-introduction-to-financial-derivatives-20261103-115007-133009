"""Experiment 4: Dupire local volatility surface (Chapter 9).

Route A: finite-difference Dupire on a constant-vol BS call surface.
Special case (Exercise 9.1): term-structure-only sigma(T) and integral identity.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.black_scholes import bs_call
from src.config import (
    DUPIRE_K_MAX_FRAC,
    DUPIRE_K_MIN_FRAC,
    DUPIRE_N_K,
    DUPIRE_N_T,
    DUPIRE_R,
    DUPIRE_S0,
    DUPIRE_SVI_A,
    DUPIRE_T_MAX,
    DUPIRE_T_MIN,
)
from src.dupire import (
    dupire_fd,
    dupire_term_structure,
    verify_integral_identity,
)

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Build grids and constant-vol call surface (Route A)
# ---------------------------------------------------------------------------

T_grid = np.linspace(DUPIRE_T_MIN, DUPIRE_T_MAX, DUPIRE_N_T)
K_grid = np.linspace(DUPIRE_K_MIN_FRAC * DUPIRE_S0, DUPIRE_K_MAX_FRAC * DUPIRE_S0, DUPIRE_N_K)

# Constant-vol call surface using DUPIRE_SVI_A as variance (sigma = sqrt(a))
sigma_const = float(DUPIRE_SVI_A**0.5)
C = np.array([
    [bs_call(0, DUPIRE_S0, sigma_const, T_grid[i], K_grid[j], DUPIRE_R)
     for j in range(DUPIRE_N_K)]
    for i in range(DUPIRE_N_T)
])

# ---------------------------------------------------------------------------
# Step 2 (Route A): Dupire finite-difference local variance
# ---------------------------------------------------------------------------

f2_fd = dupire_fd(C, T_grid, K_grid)

print("=== Route A: Dupire FD ===")
print(f"  f2_fd shape: {f2_fd.shape}")
print(f"  f2_fd (nanmin, nanmax): {float(np.nanmin(f2_fd)):.6f}, {float(np.nanmax(f2_fd)):.6f}")

# Heatmap of local-variance surface
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(
    f2_fd,
    aspect="auto",
    origin="lower",
    extent=[K_grid[1], K_grid[-2], T_grid[1], T_grid[-2]],
)
fig.colorbar(im, ax=ax, label="Local variance f²(T,K)")
ax.set_xlabel("Strike K")
ax.set_ylabel("Maturity T")
ax.set_title("Dupire local variance (finite-difference route)")
fig.tight_layout()
fig.savefig("results/exp4_dupire_fd.png", dpi=100)
plt.close(fig)

# ---------------------------------------------------------------------------
# Step 3 (Exercise 9.1): Term-structure-only implied vol
# ---------------------------------------------------------------------------

print("\n=== Exercise 9.1: Term-structure sigma(T) ===")
T_sp = np.linspace(0.5, 2.0, 10)
sigma_T = 0.15 + 0.05 * np.exp(-T_sp)

# f2 at K = DUPIRE_S0
f2_ts = dupire_term_structure(DUPIRE_S0, T_sp, sigma_T, K=DUPIRE_S0, r=DUPIRE_R)
print("  f2_ts (local variance, interior nodes):")
print(" ", f2_ts)

# Verify K-independence: compute at a second strike and compare
f2_ts_alt = dupire_term_structure(DUPIRE_S0, T_sp, sigma_T, K=DUPIRE_S0 * 1.1, r=DUPIRE_R)
print("  f2_ts at K=1.1*S0 (should be similar):")
print(" ", f2_ts_alt)

# ---------------------------------------------------------------------------
# Step 4: Verify integral identity
# ---------------------------------------------------------------------------

errors_id = verify_integral_identity(T_sp, f2_ts, sigma_T)
print("\n  Integral identity errors:")
print(" ", errors_id)
max_err = float(np.nanmax(errors_id))
print(f"  Max error: {max_err:.6f}")
assert max_err < 0.05, f"Integral identity max error {max_err} >= 0.05"

# Plot f2_ts vs T (interior maturities T_sp[1:-1])
interior_T = T_sp[1:-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(interior_T, f2_ts, marker="o", label="f²(T, K=S₀)")
ax.set_xlabel("Maturity T")
ax.set_ylabel("Local variance f²")
ax.set_title("Dupire local variance – term-structure-only implied vol (Exercise 9.1)")
ax.legend()
fig.tight_layout()
fig.savefig("results/exp4_local_vol_ts.png", dpi=100)
plt.close(fig)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results_dict = {
    "f2_fd_min": float(np.nanmin(f2_fd)),
    "f2_fd_max": float(np.nanmax(f2_fd)),
    "f2_ts_mean": float(np.nanmean(f2_ts)),
    "integral_identity_max_error": max_err,
}
with open("results/exp4_results.json", "w") as fh:
    json.dump(results_dict, fh, indent=2)

print("\nResults saved to results/exp4_results.json")
print("Plots saved to results/exp4_dupire_fd.png and results/exp4_local_vol_ts.png")
