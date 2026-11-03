"""Tests for Dupire local volatility (Chapter 9)."""

import numpy as np
import pytest

from src.black_scholes import bs_call
from src.dupire import (
    dupire_fd,
    dupire_term_structure,
    svi_implied_vol,
    verify_integral_identity,
)


def _make_constant_vol_surface(
    sigma: float = 0.20,
    n_T: int = 6,
    n_K: int = 8,
    S0: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a constant-vol BS call surface for tests."""
    T_grid = np.linspace(0.25, 2.0, n_T)
    K_grid = np.linspace(70.0, 130.0, n_K)
    C = np.array([
        [bs_call(0.0, S0, sigma, T, K, 0.0) for K in K_grid]
        for T in T_grid
    ])
    return C, T_grid, K_grid


def test_dupire_fd_positive() -> None:
    """Local variance f² from finite differences should be >= 0 where defined."""
    C, T_grid, K_grid = _make_constant_vol_surface()
    f2 = dupire_fd(C, T_grid, K_grid)
    # Interior nodes only; NaN allowed (ill-conditioned edges)
    valid = f2[~np.isnan(f2)]
    assert len(valid) > 0, "No valid (non-NaN) nodes returned."
    assert float(np.min(valid)) >= -1e-6, f"Negative local variance found: {np.min(valid)}"


def test_dupire_term_structure_k_independent() -> None:
    """f² depends only on T when sigma(T) is K-independent, so two strikes give equal results."""
    T_sp = np.linspace(0.5, 2.0, 8)
    sigma_T = 0.15 + 0.05 * np.exp(-T_sp)
    S0 = 100.0

    f2_K1 = dupire_term_structure(S0, T_sp, sigma_T, K=S0, r=0.0)
    f2_K2 = dupire_term_structure(S0, T_sp, sigma_T, K=S0 * 1.1, r=0.0)

    # Both arrays should agree to within finite-difference numerical noise
    # (eps = K*1e-4 in dupire_term_structure gives absolute errors ~O(1e-4))
    diff = float(np.nanmax(np.abs(f2_K1 - f2_K2)))
    assert diff < 5e-4, f"K-independence violated; max diff={diff}"


def test_integral_identity_accuracy() -> None:
    """verify_integral_identity should return small errors (< 0.1)."""
    T_sp = np.linspace(0.5, 2.0, 10)
    sigma_T = 0.15 + 0.05 * np.exp(-T_sp)
    S0 = 100.0

    f2_ts = dupire_term_structure(S0, T_sp, sigma_T, K=S0, r=0.0)
    errors = verify_integral_identity(T_sp, f2_ts, sigma_T)

    valid_errors = errors[~np.isnan(errors)]
    assert len(valid_errors) > 0, "All identity errors are NaN."
    max_err = float(np.max(valid_errors))
    assert max_err < 0.10, f"Integral identity max error {max_err:.4f} >= 0.10"


def test_svi_positive_variance() -> None:
    """SVI implied vol should be real and positive for typical parameters."""
    # Typical SVI params: a=0.04, b=0.10, rho=-0.3, m=0.0, xi=0.15
    k_vals = np.linspace(-0.5, 0.5, 20)
    x = 0.0  # log(S0) with S0=1
    vols = svi_implied_vol(k_vals, x, a=0.04, b=0.10, rho=-0.3, m=0.0, xi=0.15)
    assert np.all(np.isfinite(vols)), "SVI returned non-finite values."
    assert np.all(vols > 0), f"SVI returned non-positive vol; min={np.min(vols)}"
