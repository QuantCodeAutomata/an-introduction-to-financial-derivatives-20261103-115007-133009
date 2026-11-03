"""Experiment 2: Black-Scholes pricing, implied vol, and power-claim hedging.

Run with: python -m exp.exp2
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.black_scholes import (
    bs_call,
    bs_put,
    power_claim_delta,
    power_claim_value,
)
from src.config import (
    BS_B0,
    BS_M,
    BS_MU,
    BS_N,
    BS_P,
    BS_R,
    BS_S0,
    BS_SIGMA,
    BS_T,
    BS_VANILLA_K,
    BS_VANILLA_R,
    BS_VANILLA_S0,
    BS_VANILLA_SIGMA,
    BS_VANILLA_T,
)
from src.implied_vol import implied_vol_call
from src.monte_carlo import (
    compute_hedging_errors,
    replicate_power_claim,
    simulate_bond_path,
    simulate_gbm_paths,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_put_call_parity() -> float:
    """Verify put-call parity: C - P = S - K*exp(-r*T)."""
    S = BS_VANILLA_S0
    K = BS_VANILLA_K
    sigma = BS_VANILLA_SIGMA
    r = BS_VANILLA_R
    T = BS_VANILLA_T
    t = 0.0

    C = bs_call(t, S, sigma, T, K, r)
    P = bs_put(t, S, sigma, T, K, r)
    lhs = C - P
    rhs = S - K * math.exp(-r * T)
    error = abs(lhs - rhs)
    assert error < 1e-10, f"Put-call parity error {error:.2e} exceeds 1e-10"
    print(f"[2a] Put-call parity error = {error:.2e}  (< 1e-10)")
    return error


def run_implied_vol_roundtrip() -> float:
    """Generate BS call price then recover sigma via implied_vol_call."""
    S = BS_VANILLA_S0
    K = BS_VANILLA_K
    sigma = BS_VANILLA_SIGMA
    r = BS_VANILLA_R
    T = BS_VANILLA_T
    t = 0.0

    price = bs_call(t, S, sigma, T, K, r)
    sigma_rec = implied_vol_call(price, t, S, T, K, r)
    assert abs(sigma_rec - sigma) < 1e-6, (
        f"Implied vol round-trip error {abs(sigma_rec - sigma):.2e} exceeds 1e-6"
    )
    print(f"[2b] Implied vol recovered = {sigma_rec:.8f}  (input {sigma})")
    return sigma_rec


def run_power_claim() -> tuple[float, float]:
    """Compute power-claim price and delta at t=0."""
    V0 = power_claim_value(0.0, BS_S0, BS_SIGMA, BS_R, BS_T, BS_P)
    Delta0 = power_claim_delta(0.0, BS_S0, BS_SIGMA, BS_R, BS_T, BS_P)
    print(f"[2c] Power claim V0 = {V0:.6f},  Delta0 = {Delta0:.6f}")
    return V0, Delta0


def run_monte_carlo_hedging() -> np.ndarray:
    """Compute terminal hedging errors over M paths."""
    rng = np.random.default_rng(42)
    errors = compute_hedging_errors(
        BS_S0, BS_MU, BS_SIGMA, BS_R, BS_T, BS_P, BS_N, BS_M, rng=rng
    )
    print(
        f"[2d] Hedging errors: mean = {errors.mean():.6f},  std = {errors.std():.6f}"
    )
    return errors


def plot_vx_path() -> None:
    """Plot one sample V_t and X_t path and save to results."""
    rng = np.random.default_rng(0)
    S_paths, t_grid = simulate_gbm_paths(BS_S0, BS_MU, BS_SIGMA, BS_T, BS_N, 1, rng=rng)
    B_path = simulate_bond_path(BS_B0, BS_R, t_grid)
    V_path, X_path = replicate_power_claim(
        S_paths[0], B_path, t_grid, BS_SIGMA, BS_R, BS_T, BS_P
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_grid, V_path, label=r"$V_t$ (exact)", linewidth=1.5)
    ax.plot(t_grid, X_path, label=r"$X_t$ (portfolio)", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Value")
    ax.set_title(r"Power claim replication: $V_t$ vs $X_t$ (one path)")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp2_vx_path.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"[2e] V/X path plot saved to {out_path}")


def plot_hedging_error_hist(errors: np.ndarray) -> None:
    """Plot histogram of terminal hedging errors X_T - V_T."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=40, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero error")
    ax.set_xlabel(r"$X_T - V_T$")
    ax.set_ylabel("Frequency")
    ax.set_title(r"Histogram of hedging errors $X_T - V_T$")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp2_hedging_error_hist.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"[2f] Hedging error histogram saved to {out_path}")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    parity_error = run_put_call_parity()
    sigma_rec = run_implied_vol_roundtrip()
    V0, Delta0 = run_power_claim()
    errors = run_monte_carlo_hedging()
    plot_vx_path()
    plot_hedging_error_hist(errors)

    results = {
        "parity_error": parity_error,
        "implied_vol_recovered": sigma_rec,
        "power_claim_V0": V0,
        "power_claim_Delta0": Delta0,
        "hedging_error_mean": float(errors.mean()),
        "hedging_error_std": float(errors.std()),
    }

    json_path = os.path.join(RESULTS_DIR, "exp2_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {json_path}")

    print("\n=== Experiment 2 Summary ===")
    print(f"{'Quantity':<28} {'Value':>16}")
    print("-" * 46)
    for key, val in results.items():
        print(f"{key:<28} {val:>16.8f}")


if __name__ == "__main__":
    main()
