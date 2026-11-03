"""Experiment 3: Volatility mismatch delta-hedging and variance-swap replication.

Part A: Section 8.1 – P&L of a delta-hedged short call under volatility mismatch.
Part B: Section 8.2 – Variance-swap fair strike via MC and static replication.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    VARSWAP_K_MAX_FRAC,
    VARSWAP_K_MIN_FRAC,
    VARSWAP_N_STRIKES,
    VARSWAP_S0,
    VARSWAP_SIGMA,
    VARSWAP_T,
    VOL_MISMATCH_K,
    VOL_MISMATCH_M,
    VOL_MISMATCH_MU,
    VOL_MISMATCH_N,
    VOL_MISMATCH_S0,
    VOL_MISMATCH_SIGMA1,
    VOL_MISMATCH_SIGMA2,
    VOL_MISMATCH_T,
)
from src.variance_swap import (
    simulate_vol_mismatch,
    varswap_fair_strike_mc,
    varswap_fair_strike_static,
)

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Part A: Volatility mismatch delta-hedging (Section 8.1)
# ---------------------------------------------------------------------------

print("=== Part A: Volatility mismatch ===")
result = simulate_vol_mismatch(
    VOL_MISMATCH_S0,
    VOL_MISMATCH_K,
    VOL_MISMATCH_T,
    VOL_MISMATCH_SIGMA1,
    VOL_MISMATCH_SIGMA2,
    VOL_MISMATCH_MU,
    VOL_MISMATCH_N,
    VOL_MISMATCH_M,
    r=0.0,
    rng=np.random.default_rng(42),
)

errors = result["errors"]
pnl_identity = result["pnl_identity"]
one_path_S = result["one_path_S"]
one_path_X = result["one_path_X"]
one_path_C = result["one_path_C"]
t_grid = result["t_grid"]

# Plot one path of X (replication portfolio) and C (option value) vs time
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_grid, one_path_X, label="Portfolio X_t")
ax.plot(t_grid, one_path_C, label="BS Call C_t (σ₂)", linestyle="--")
ax.set_xlabel("Time t")
ax.set_ylabel("Value")
ax.set_title("Vol-mismatch: replication portfolio vs option value (one path)")
ax.legend()
fig.tight_layout()
fig.savefig("results/exp3_vol_mismatch_path.png", dpi=100)
plt.close(fig)

# Histogram of hedging errors X_T – (S_T – K)+
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
ax.axvline(float(np.mean(errors)), color="red", linestyle="--", label=f"mean={np.mean(errors):.4f}")
ax.set_xlabel("X_T – (S_T – K)+")
ax.set_ylabel("Frequency")
ax.set_title("Vol-mismatch hedging error distribution")
ax.legend()
fig.tight_layout()
fig.savefig("results/exp3_vol_mismatch_hist.png", dpi=100)
plt.close(fig)

err_mean = float(np.mean(errors))
err_min = float(np.min(errors))
err_std = float(np.std(errors))
print(f"  errors mean={err_mean:.6f}  min={err_min:.6f}  std={err_std:.6f}")
# sigma2 > sigma1, so the hedger over-hedges and earns a positive P&L
assert err_mean > 0, f"Expected positive mean P&L (sigma2>sigma1), got {err_mean}"

pnl_id_mean = float(np.mean(pnl_identity))
print(f"  pnl_identity mean={pnl_id_mean:.6f}")
# Continuous P&L approximation should be close to the discrete hedging error
rel_diff = abs(pnl_id_mean - err_mean) / (abs(err_mean) + 1e-12)
assert rel_diff < 0.20, f"pnl_identity mean not close to errors mean: {pnl_id_mean} vs {err_mean}"

# ---------------------------------------------------------------------------
# Part B: Variance swap (Section 8.2)
# ---------------------------------------------------------------------------

print("\n=== Part B: Variance swap ===")

K_mc = varswap_fair_strike_mc(
    VARSWAP_S0,
    VARSWAP_SIGMA,
    VARSWAP_T,
    N=1000,
    M=10000,
    rng=np.random.default_rng(123),
)

K_static = varswap_fair_strike_static(
    VARSWAP_S0,
    VARSWAP_T,
    lambda k: VARSWAP_SIGMA,  # constant-vol smile
    K_min_frac=VARSWAP_K_MIN_FRAC,
    K_max_frac=VARSWAP_K_MAX_FRAC,
    n_strikes=VARSWAP_N_STRIKES,
    r=0.0,
)

# Analytic fair strike for constant vol with r=0: K = sigma^2 * T (page 136)
K_theory = VARSWAP_SIGMA**2 * VARSWAP_T

print(f"  K_mc     = {K_mc:.6f}")
print(f"  K_static = {K_static:.6f}")
print(f"  K_theory = {K_theory:.6f}")

tol = 0.10  # within 10%
assert abs(K_mc - K_theory) / K_theory < tol, f"K_mc={K_mc} not within 10% of K_theory={K_theory}"
assert abs(K_static - K_theory) / K_theory < tol, f"K_static={K_static} not within 10% of K_theory={K_theory}"

# Save JSON results
results_dict = {
    "mismatch_error_mean": err_mean,
    "mismatch_error_std": err_std,
    "varswap_K_mc": float(K_mc),
    "varswap_K_static": float(K_static),
    "varswap_K_theory": float(K_theory),
}
with open("results/exp3_results.json", "w") as fh:
    json.dump(results_dict, fh, indent=2)

print("\nResults saved to results/exp3_results.json")
print("Plots saved to results/exp3_vol_mismatch_path.png and results/exp3_vol_mismatch_hist.png")
