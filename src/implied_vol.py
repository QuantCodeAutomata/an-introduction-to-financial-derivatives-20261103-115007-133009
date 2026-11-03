"""Implied volatility extraction via 1-D root finding.

The implied volatility sigma(T, K) is defined as the unique positive solution
of C^BS(t, S_t; sigma, T, K) = C_market  (Eq. 6.21, page 115).

Uses Brent's method (scipy.optimize.brentq) on the interval [1e-8, 5].
"""

from __future__ import annotations

import math
from typing import Callable

from scipy.optimize import brentq

from src.black_scholes import bs_call, bs_put


def implied_vol_call(
    market_price: float,
    t: float,
    S: float,
    T: float,
    K: float,
    r: float,
    *,
    tol: float = 1e-9,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
) -> float:
    """Extract implied volatility from a European call market price (Eq. 6.21).

    Parameters
    ----------
    market_price : observed call price
    t            : current time
    S            : current stock price
    T, K         : maturity and strike
    r            : risk-free rate
    tol          : root-finding tolerance
    sigma_lo/hi  : bracketing interval for sigma

    Returns
    -------
    Implied volatility (positive float).

    Raises
    ------
    ValueError if market_price is outside no-arbitrage bounds or root not found.
    """
    tau = T - t
    # No-arbitrage bounds for a call
    lb = max(S - K * math.exp(-r * tau), 0.0)
    ub = S
    if market_price < lb - tol or market_price > ub + tol:
        raise ValueError(
            f"Call price {market_price:.6f} outside no-arbitrage bounds "
            f"[{lb:.6f}, {ub:.6f}]."
        )

    def objective(sigma: float) -> float:
        return bs_call(t, S, sigma, T, K, r) - market_price

    f_lo = objective(sigma_lo)
    f_hi = objective(sigma_hi)
    if f_lo > 0:
        raise ValueError(
            "Cannot bracket implied vol: BS price at sigma_lo already exceeds market price."
        )
    if f_hi < 0:
        raise ValueError(
            "Cannot bracket implied vol: BS price at sigma_hi still below market price. "
            "Try increasing sigma_hi."
        )
    return brentq(objective, sigma_lo, sigma_hi, xtol=tol, full_output=False)


def implied_vol_put(
    market_price: float,
    t: float,
    S: float,
    T: float,
    K: float,
    r: float,
    *,
    tol: float = 1e-9,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
) -> float:
    """Extract implied volatility from a European put market price.

    Analogous to implied_vol_call using the put formula P^BS (Eq. 6.19).
    """
    tau = T - t
    lb = max(K * math.exp(-r * tau) - S, 0.0)
    ub = K * math.exp(-r * tau)
    if market_price < lb - tol or market_price > ub + tol:
        raise ValueError(
            f"Put price {market_price:.6f} outside no-arbitrage bounds "
            f"[{lb:.6f}, {ub:.6f}]."
        )

    def objective(sigma: float) -> float:
        return bs_put(t, S, sigma, T, K, r) - market_price

    f_lo = objective(sigma_lo)
    f_hi = objective(sigma_hi)
    if f_lo > 0:
        raise ValueError("Cannot bracket: BS put price at sigma_lo exceeds market price.")
    if f_hi < 0:
        raise ValueError("Cannot bracket: BS put price at sigma_hi below market price.")
    return brentq(objective, sigma_lo, sigma_hi, xtol=tol, full_output=False)
