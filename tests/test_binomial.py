"""Tests for binomial tree pricing (Chapter 3)."""

import math

import numpy as np
import pytest

from src.binomial import (
    calibrate_crr,
    call_payoff,
    price_american_binomial,
    price_european_binomial,
    price_european_onestep,
    put_payoff,
)


def test_onestep_call_lecture_example() -> None:
    """One-step call price matches the lecture example (V0 = 5)."""
    result = price_european_onestep(100.0, 1.1, 0.9, 0.0, call_payoff(100.0))
    assert abs(result["V0"] - 5.0) < 1e-10


def test_onestep_replication_equals_risk_neutral() -> None:
    """price_european_onestep should return without raising (internal assertion holds)."""
    # The function internally asserts replication == risk-neutral; no exception means pass.
    result = price_european_onestep(80.0, 1.2, 0.85, 0.02, call_payoff(85.0))
    assert "V0" in result
    assert "Delta0" in result


def test_twostep_call_lecture_example() -> None:
    """Two-step call with r=0 discrete rate matches lecture value (V0 = 5.25)."""
    S0, u, d, r, N, K = 100.0, 1.1, 0.9, 0.0, 2, 100.0
    disc = 1.0 / (1.0 + r)          # = 1.0
    p_tilde = ((1.0 + r) - d) / (u - d)  # = 0.5
    q_tilde = 1.0 - p_tilde
    result = price_european_binomial(S0, u, d, disc, p_tilde, q_tilde, N, call_payoff(K))
    assert abs(result["V0"] - 5.25) < 1e-10


def test_european_put_calibrated() -> None:
    """CRR-calibrated European put converges to the textbook values (Exercise 3.4)."""
    crr = calibrate_crr(0.10, 0.05, 2.0, 50)
    result = price_european_binomial(
        50.0,
        crr["u"], crr["d"], crr["disc"],
        crr["p_tilde"], crr["q_tilde"],
        50,
        put_payoff(55.0),
    )
    assert abs(result["V0"] - 2.6955) < 0.01, f"V0={result['V0']}"
    assert abs(result["Delta0"] - (-0.4623)) < 0.01, f"Delta0={result['Delta0']}"


def test_american_ge_european() -> None:
    """American put price must be at least as large as the corresponding European put."""
    crr = calibrate_crr(0.20, 0.05, 1.0, 50)
    kwargs = dict(
        S0=100.0,
        u=crr["u"], d=crr["d"], disc=crr["disc"],
        p_tilde=crr["p_tilde"], q_tilde=crr["q_tilde"],
        N=50,
        payoff=put_payoff(100.0),
    )
    eur = price_european_binomial(**kwargs)
    am = price_american_binomial(**kwargs)
    assert am["V0"] >= eur["V0"] - 1e-10


def test_no_arbitrage_violation() -> None:
    """calibrate_crr raises ValueError when d >= exp(r*dt), violating no-arbitrage."""
    # With sigma very small, u ≈ d ≈ 1, and for r=0.10, exp(r*dt) > u is possible
    # Force violation by setting sigma nearly zero so d > exp(r*dt) fails
    # Actually with sigma→0, u→1 and d→1 but exp(r*dt)>1 for r>0, so p_tilde→∞
    with pytest.raises(ValueError):
        calibrate_crr(sigma=1e-8, r=0.10, T=1.0, N=1)


def test_risk_neutral_prob_range() -> None:
    """For valid CRR parameters, p_tilde must lie strictly in (0, 1)."""
    crr = calibrate_crr(0.20, 0.05, 1.0, 10)
    assert 0.0 < crr["p_tilde"] < 1.0
    assert 0.0 < crr["q_tilde"] < 1.0
    assert abs(crr["p_tilde"] + crr["q_tilde"] - 1.0) < 1e-14


def test_put_call_parity_binomial() -> None:
    """For European options, C - P ≈ S0 - K/(1+r)^N (discrete put-call parity)."""
    S0, u, d, r, N, K = 100.0, 1.1, 0.9, 0.0, 4, 100.0
    disc = 1.0 / (1.0 + r)
    p_tilde = ((1.0 + r) - d) / (u - d)
    q_tilde = 1.0 - p_tilde
    common = dict(S0=S0, u=u, d=d, disc=disc, p_tilde=p_tilde, q_tilde=q_tilde, N=N)
    call_res = price_european_binomial(**common, payoff=call_payoff(K))
    put_res = price_european_binomial(**common, payoff=put_payoff(K))
    lhs = call_res["V0"] - put_res["V0"]
    rhs = S0 - K * disc**N        # K discounted N steps
    assert abs(lhs - rhs) < 1e-8, f"Parity error: {abs(lhs - rhs):.2e}"
