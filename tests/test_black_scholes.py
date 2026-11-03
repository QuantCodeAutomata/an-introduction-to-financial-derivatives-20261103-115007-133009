"""Tests for Black-Scholes pricing and implied vol inversion (Chapter 6)."""

import math

import numpy as np
import pytest

from src.black_scholes import (
    bs_call,
    bs_delta_call,
    bs_delta_put,
    bs_gamma,
    bs_put,
    power_claim_delta,
    power_claim_value,
)
from src.implied_vol import implied_vol_call, implied_vol_put


# ---------------------------------------------------------------------------
# Fixtures / parameter sets used across multiple tests
# ---------------------------------------------------------------------------

_CASES = [
    # (S, K, r, sigma, T)
    (100.0, 100.0, 0.05, 0.20, 1.0),
    (110.0,  95.0, 0.03, 0.15, 0.5),
    ( 80.0, 100.0, 0.01, 0.30, 2.0),
]


# ---------------------------------------------------------------------------
# Put-call parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K,r,sigma,T", _CASES)
def test_put_call_parity(S: float, K: float, r: float, sigma: float, T: float) -> None:
    """C - P = S - K*exp(-r*T) (Black-Scholes put-call parity)."""
    t = 0.0
    C = bs_call(t, S, sigma, T, K, r)
    P = bs_put(t, S, sigma, T, K, r)
    lhs = C - P
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10, f"Parity error {abs(lhs - rhs):.2e}"


# ---------------------------------------------------------------------------
# Delta ranges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K,r,sigma,T", _CASES)
def test_call_delta_range(S: float, K: float, r: float, sigma: float, T: float) -> None:
    """Delta of a European call must lie in (0, 1)."""
    delta = bs_delta_call(0.0, S, sigma, T, K, r)
    assert 0.0 < delta < 1.0, f"Call delta {delta} out of range"


@pytest.mark.parametrize("S,K,r,sigma,T", _CASES)
def test_put_delta_range(S: float, K: float, r: float, sigma: float, T: float) -> None:
    """Delta of a European put must lie in (-1, 0)."""
    delta = bs_delta_put(0.0, S, sigma, T, K, r)
    assert -1.0 < delta < 0.0, f"Put delta {delta} out of range"


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K,r,sigma,T", _CASES)
def test_gamma_positive(S: float, K: float, r: float, sigma: float, T: float) -> None:
    """Gamma must be strictly positive for tau > 0."""
    gamma = bs_gamma(0.0, S, sigma, T, K, r)
    assert gamma > 0.0, f"Gamma {gamma} not positive"


# ---------------------------------------------------------------------------
# Monotonicity in sigma
# ---------------------------------------------------------------------------

def test_call_monotone_in_sigma() -> None:
    """BS call price must be strictly increasing in sigma."""
    sigmas = [0.10, 0.20, 0.30, 0.40]
    prices = [bs_call(0.0, 100.0, s, 1.0, 100.0, 0.05) for s in sigmas]
    for lo, hi in zip(prices, prices[1:]):
        assert hi > lo, f"Call price not increasing: {lo:.4f} -> {hi:.4f}"


# ---------------------------------------------------------------------------
# Implied vol round-trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K,r,sigma,T", _CASES)
def test_implied_vol_roundtrip_call(
    S: float, K: float, r: float, sigma: float, T: float
) -> None:
    """Implied vol recovered from BS call price must match input sigma to 1e-6."""
    price = bs_call(0.0, S, sigma, T, K, r)
    sigma_rec = implied_vol_call(price, 0.0, S, T, K, r)
    assert abs(sigma_rec - sigma) < 1e-6, f"Round-trip error {abs(sigma_rec - sigma):.2e}"


@pytest.mark.parametrize("S,K,r,sigma,T", _CASES)
def test_implied_vol_roundtrip_put(
    S: float, K: float, r: float, sigma: float, T: float
) -> None:
    """Implied vol recovered from BS put price must match input sigma to 1e-6."""
    price = bs_put(0.0, S, sigma, T, K, r)
    sigma_rec = implied_vol_put(price, 0.0, S, T, K, r)
    assert abs(sigma_rec - sigma) < 1e-6, f"Round-trip error {abs(sigma_rec - sigma):.2e}"


# ---------------------------------------------------------------------------
# Power claim
# ---------------------------------------------------------------------------

def test_power_claim_terminal() -> None:
    """At t=T the power-claim value must equal S^p exactly."""
    for S, p in [(1.0, 2.0), (2.5, -1.0), (0.8, 3.0)]:
        V = power_claim_value(1.0, S, 0.2, 0.05, 1.0, p)
        assert abs(V - S**p) < 1e-12, f"Terminal mismatch for S={S}, p={p}: {V} vs {S**p}"


def test_power_claim_delta_sign() -> None:
    """For p=2, Delta = 2*S * exp(...) > 0."""
    delta = power_claim_delta(0.0, 1.0, 0.2, 0.05, 1.0, 2.0)
    assert delta > 0.0, f"Expected positive Delta for p=2, got {delta}"


# ---------------------------------------------------------------------------
# Call at expiry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K", [(110.0, 100.0), (90.0, 100.0), (100.0, 100.0)])
def test_call_at_expiry(S: float, K: float) -> None:
    """bs_call(T, S, sigma, T, K, r) must equal max(S - K, 0)."""
    val = bs_call(1.0, S, 0.2, 1.0, K, 0.05)
    expected = max(S - K, 0.0)
    assert abs(val - expected) < 1e-12, f"Expiry mismatch: {val} vs {expected}"
