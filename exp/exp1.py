"""Experiment 1: Binomial tree pricing of European and American options.

Run with: python -m exp.exp1
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.binomial import (
    calibrate_crr,
    call_payoff,
    price_american_binomial,
    price_european_binomial,
    price_european_onestep,
    put_payoff,
)
from src.config import (
    BINOM_AM_K,
    BINOM_AM_N,
    BINOM_AM_R,
    BINOM_AM_S0,
    BINOM_AM_SIGMA,
    BINOM_AM_T,
    BINOM_EUR_K,
    BINOM_EUR_N,
    BINOM_EUR_R,
    BINOM_EUR_S0,
    BINOM_EUR_SIGMA,
    BINOM_EUR_T,
    BINOM_ONESTEP_D,
    BINOM_ONESTEP_K,
    BINOM_ONESTEP_R,
    BINOM_ONESTEP_S0,
    BINOM_ONESTEP_U,
    BINOM_TWOSTEP_D,
    BINOM_TWOSTEP_K,
    BINOM_TWOSTEP_N,
    BINOM_TWOSTEP_R,
    BINOM_TWOSTEP_S0,
    BINOM_TWOSTEP_U,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_onestep_call() -> float:
    """One-step European call using simple (not continuous) rate r=0."""
    S0 = BINOM_ONESTEP_S0
    u = BINOM_ONESTEP_U
    d = BINOM_ONESTEP_D
    r = BINOM_ONESTEP_R
    K = BINOM_ONESTEP_K
    result = price_european_onestep(S0, u, d, r, call_payoff(K))
    print(f"[1a] One-step call V0 = {result['V0']:.4f}  (expected 5.0)")
    return result["V0"]


def run_twostep_call() -> float:
    """Two-step European call using discrete rate formula (Section 3.2.1)."""
    S0 = BINOM_TWOSTEP_S0
    u = BINOM_TWOSTEP_U
    d = BINOM_TWOSTEP_D
    r = BINOM_TWOSTEP_R
    N = BINOM_TWOSTEP_N
    K = BINOM_TWOSTEP_K

    # Discrete-rate no-arbitrage parameters (not CRR continuous-rate calibration)
    disc_step = 1.0 / (1.0 + r)           # = 1.0 for r=0
    p_tilde = ((1.0 + r) - d) / (u - d)   # = 0.5 for these values
    q_tilde = 1.0 - p_tilde

    result = price_european_binomial(S0, u, d, disc_step, p_tilde, q_tilde, N, call_payoff(K))
    print(f"[1b] Two-step call V0 = {result['V0']:.4f}  (expected 5.25)")
    return result["V0"]


def run_european_put() -> dict:
    """European put priced on CRR-calibrated tree (Exercise 3.4)."""
    crr = calibrate_crr(BINOM_EUR_SIGMA, BINOM_EUR_R, BINOM_EUR_T, BINOM_EUR_N)
    result = price_european_binomial(
        BINOM_EUR_S0,
        crr["u"], crr["d"], crr["disc"],
        crr["p_tilde"], crr["q_tilde"],
        BINOM_EUR_N,
        put_payoff(BINOM_EUR_K),
    )
    print(
        f"[1c] European put V0 = {result['V0']:.4f}  (expected ≈2.6955), "
        f"Delta0 = {result['Delta0']:.4f}  (expected ≈-0.4623)"
    )
    return result


def run_american_put() -> dict:
    """American put priced on CRR-calibrated tree (Exercise 3.5)."""
    crr = calibrate_crr(BINOM_AM_SIGMA, BINOM_AM_R, BINOM_AM_T, BINOM_AM_N)
    result = price_american_binomial(
        BINOM_AM_S0,
        crr["u"], crr["d"], crr["disc"],
        crr["p_tilde"], crr["q_tilde"],
        BINOM_AM_N,
        put_payoff(BINOM_AM_K),
    )
    print(f"[1d] American put V0 = {result['V0']:.4f}, Delta0 = {result['Delta0']:.4f}")
    return result


def save_bar_chart(eur_v0: float, am_v0: float) -> None:
    """Save bar chart comparing European vs American put prices."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["European Put", "American Put"]
    values = [eur_v0, am_v0]
    bars = ax.bar(labels, values, color=["steelblue", "darkorange"], width=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylabel("Option Price ($)")
    ax.set_title("Binomial Put Prices: European vs American")
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp1_binomial_prices.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"[1f] Bar chart saved to {out_path}")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Sub-experiments
    onestep_v0 = run_onestep_call()
    twostep_v0 = run_twostep_call()
    eur_result = run_european_put()
    am_result = run_american_put()

    # Verify American >= European for same put parameters
    assert am_result["V0"] >= eur_result["V0"] - 1e-10, (
        f"American put {am_result['V0']:.6f} < European put {eur_result['V0']:.6f}"
    )
    print("[1e] Assertion passed: American put V0 >= European put V0")

    save_bar_chart(eur_result["V0"], am_result["V0"])

    # Collect results
    results = {
        "onestep_call_V0": onestep_v0,
        "twostep_call_V0": twostep_v0,
        "european_put_V0": eur_result["V0"],
        "european_put_Delta0": eur_result["Delta0"],
        "american_put_V0": am_result["V0"],
        "american_put_Delta0": am_result["Delta0"],
    }

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "exp1_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {json_path}")

    # Summary table
    print("\n=== Experiment 1 Summary ===")
    print(f"{'Quantity':<28} {'Value':>12}")
    print("-" * 42)
    for key, val in results.items():
        print(f"{key:<28} {val:>12.6f}")


if __name__ == "__main__":
    main()
