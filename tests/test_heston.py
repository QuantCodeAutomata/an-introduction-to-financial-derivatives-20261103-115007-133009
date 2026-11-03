"""Tests for Heston model pricing (Chapter 10)."""

import math

import numpy as np
import pytest

from src.heston import HestonParams, heston_call, heston_implied_vol


def _base_params() -> HestonParams:
    """Reasonable Heston params for tests."""
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, delta=0.30, rho=-0.70, r=0.0)


def test_heston_call_positive() -> None:
    """Call price > 0 for in-the-money option."""
    params = _base_params()
    # Deep in the money
    price = heston_call(110.0, 100.0, 1.0, params)
    assert price > 0.0, f"Expected positive call price, got {price}"


def test_heston_call_lower_bound() -> None:
    """Call price >= max(S - K*exp(-r*T), 0)."""
    params = _base_params()
    S, K, T = 100.0, 90.0, 1.0
    price = heston_call(S, K, T, params)
    lb = max(S - K * math.exp(-params.r * T), 0.0)
    assert price >= lb - 1e-6, f"Call price {price:.6f} below lower bound {lb:.6f}"


def test_heston_call_at_zero_vol_lim() -> None:
    """With delta→0 and v0=theta=sigma^2, Heston call approaches BS call."""
    from src.black_scholes import bs_call

    sigma = 0.20
    params = HestonParams(v0=sigma**2, kappa=2.0, theta=sigma**2, delta=1e-4, rho=0.0, r=0.0)
    S, K, T = 100.0, 100.0, 1.0
    heston_price = heston_call(S, K, T, params, n_integration=200, upper_limit=300.0)
    bs_price = bs_call(0.0, S, sigma, T, K, 0.0)
    assert abs(heston_price - bs_price) < 0.05, (
        f"Heston price {heston_price:.4f} not close to BS {bs_price:.4f}"
    )


def test_heston_implied_vol_returns_float() -> None:
    """heston_implied_vol returns a finite positive float for normal parameters."""
    params = _base_params()
    iv = heston_implied_vol(100.0, 100.0, 1.0, params)
    assert iv is not None, "Expected a float, got None."
    assert math.isfinite(iv), f"Expected finite implied vol, got {iv}"
    assert iv > 0.0, f"Expected positive implied vol, got {iv}"


def test_heston_call_increasing_in_T() -> None:
    """ATM call price increases with maturity (time value is non-negative)."""
    params = _base_params()
    S, K = 100.0, 100.0
    prices = [heston_call(S, K, T, params) for T in [0.5, 1.0, 2.0]]
    assert prices[0] < prices[1], f"Price at T=0.5 ({prices[0]:.4f}) >= T=1.0 ({prices[1]:.4f})"
    assert prices[1] < prices[2], f"Price at T=1.0 ({prices[1]:.4f}) >= T=2.0 ({prices[2]:.4f})"
