"""Tests for variance swap and vol mismatch (Chapter 8)."""

import numpy as np
import pytest

from src.variance_swap import (
    simulate_vol_mismatch,
    varswap_fair_strike_mc,
    varswap_fair_strike_static,
)

# Small grid for simulate_vol_mismatch (has a Python inner loop – slow per path)
N_MISMATCH = 50
M_MISMATCH = 500

# varswap_fair_strike_mc is pure NumPy; use default-sized grid for reliable statistics
N_MC = 1000
M_MC = 10000


def test_vol_mismatch_positive_pnl() -> None:
    """sigma2 > sigma1 gives positive mean P&L (E[X_T - (S_T-K)+] > 0)."""
    rng = np.random.default_rng(0)
    result = simulate_vol_mismatch(
        100.0, 100.0, 1.0, 0.15, 0.20, 0.05, N_MISMATCH, M_MISMATCH, r=0.0, rng=rng
    )
    mean_err = float(np.mean(result["errors"]))
    assert mean_err > 0.0, f"Expected positive mean P&L, got {mean_err}"


def test_vol_mismatch_zero_when_equal() -> None:
    """sigma1 == sigma2 gives P&L near zero mean."""
    rng = np.random.default_rng(0)
    result = simulate_vol_mismatch(
        100.0, 100.0, 1.0, 0.20, 0.20, 0.05, N_MISMATCH, M_MISMATCH, r=0.0, rng=rng
    )
    mean_err = float(np.mean(result["errors"]))
    # Equal vols → portfolio replicates the option; small residual from discretisation
    assert abs(mean_err) < 1.0, f"Mean P&L should be near zero, got {mean_err}"


def test_varswap_constant_vol_mc() -> None:
    """K_mc ≈ sigma^2 * T for constant vol (analytic result, page 136).

    Uses N=1000, M=10000 (function defaults) because varswap_fair_strike_mc is
    pure NumPy and runs fast; M=500 would give a standard error of ~0.018,
    making a 10% tolerance unreliable.
    """
    sigma, T = 0.20, 1.0
    K_theory = sigma**2 * T
    rng = np.random.default_rng(0)
    K_mc = varswap_fair_strike_mc(100.0, sigma, T, N=N_MC, M=M_MC, rng=rng)
    rel = abs(K_mc - K_theory) / K_theory
    assert rel < 0.10, f"K_mc={K_mc:.5f} not within 10% of K_theory={K_theory:.5f}"


def test_varswap_constant_vol_static() -> None:
    """K_static ≈ sigma^2 * T for constant vol (analytic result)."""
    sigma, T = 0.20, 1.0
    K_theory = sigma**2 * T
    K_static = varswap_fair_strike_static(
        100.0, T, lambda k: sigma, K_min_frac=0.2, K_max_frac=3.0, n_strikes=200, r=0.0
    )
    rel = abs(K_static - K_theory) / K_theory
    assert rel < 0.10, f"K_static={K_static:.5f} not within 10% of K_theory={K_theory:.5f}"


def test_pnl_identity_approx() -> None:
    """mean(errors) ≈ mean(pnl_identity) within 20%."""
    rng = np.random.default_rng(0)
    result = simulate_vol_mismatch(
        100.0, 100.0, 1.0, 0.15, 0.20, 0.05, N_MISMATCH, M_MISMATCH, r=0.0, rng=rng
    )
    mean_err = float(np.mean(result["errors"]))
    mean_pnl = float(np.mean(result["pnl_identity"]))
    rel = abs(mean_pnl - mean_err) / (abs(mean_err) + 1e-12)
    assert rel < 0.20, (
        f"pnl_identity mean ({mean_pnl:.5f}) not close to errors mean ({mean_err:.5f}); rel={rel:.3f}"
    )
