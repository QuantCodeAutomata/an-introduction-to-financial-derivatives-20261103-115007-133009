"""Black-Scholes pricing, Greeks, and power-claim formulas.

Implements the formulas from Chapter 6 of the notes:
  - European call/put prices (Eq. 6.18–6.19)
  - Delta for calls and puts (page 115)
  - Gamma (second derivative)
  - Power-claim price and Delta (Exercise 6.1, page 118)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# d± helper (Eq. 6.20)
# ---------------------------------------------------------------------------

def d_plus(t: float, S: float, sigma: float, T: float, K: float, r: float) -> float:
    """Compute d_+ from Black-Scholes formula (Eq. 6.20)."""
    tau = T - t
    if tau <= 0.0:
        return math.inf if S > K else -math.inf
    return (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))


def d_minus(t: float, S: float, sigma: float, T: float, K: float, r: float) -> float:
    """Compute d_- from Black-Scholes formula (Eq. 6.20)."""
    tau = T - t
    if tau <= 0.0:
        return math.inf if S > K else -math.inf
    return (math.log(S / K) + (r - 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))


# ---------------------------------------------------------------------------
# Call and put prices (Eq. 6.18–6.19)
# ---------------------------------------------------------------------------

def bs_call(t: float, S: float, sigma: float, T: float, K: float, r: float) -> float:
    """Black-Scholes European call price (Eq. 6.18).

    C^BS(t, S_t; sigma, T, K) = S_t * Phi(d_+) - K * exp(-r*(T-t)) * Phi(d_-)
    """
    tau = T - t
    if tau <= 0.0:
        return max(S - K, 0.0)
    dp = d_plus(t, S, sigma, T, K, r)
    dm = d_minus(t, S, sigma, T, K, r)
    return S * norm.cdf(dp) - K * math.exp(-r * tau) * norm.cdf(dm)


def bs_put(t: float, S: float, sigma: float, T: float, K: float, r: float) -> float:
    """Black-Scholes European put price (Eq. 6.19).

    P^BS(t, S_t; sigma, T, K) = K * exp(-r*(T-t)) * Phi(-d_-) - S_t * Phi(-d_+)
    """
    tau = T - t
    if tau <= 0.0:
        return max(K - S, 0.0)
    dp = d_plus(t, S, sigma, T, K, r)
    dm = d_minus(t, S, sigma, T, K, r)
    return K * math.exp(-r * tau) * norm.cdf(-dm) - S * norm.cdf(-dp)


# ---------------------------------------------------------------------------
# Delta (page 115)
# ---------------------------------------------------------------------------

def bs_delta_call(t: float, S: float, sigma: float, T: float, K: float, r: float) -> float:
    """Delta of a European call: Delta = Phi(d_+) ∈ (0, 1)."""
    tau = T - t
    if tau <= 0.0:
        return 1.0 if S > K else 0.0
    return float(norm.cdf(d_plus(t, S, sigma, T, K, r)))


def bs_delta_put(t: float, S: float, sigma: float, T: float, K: float, r: float) -> float:
    """Delta of a European put: Delta = Phi(d_+) - 1 ∈ (-1, 0)."""
    return bs_delta_call(t, S, sigma, T, K, r) - 1.0


# ---------------------------------------------------------------------------
# Gamma (second derivative w.r.t. S, same for call and put)
# ---------------------------------------------------------------------------

def bs_gamma(t: float, S: float, sigma: float, T: float, K: float, r: float) -> float:
    """Gamma of a European option: partial^2 C / partial S^2 = phi(d+) / (S*sigma*sqrt(tau))."""
    tau = T - t
    if tau <= 0.0:
        return 0.0
    dp = d_plus(t, S, sigma, T, K, r)
    return float(norm.pdf(dp)) / (S * sigma * math.sqrt(tau))


# ---------------------------------------------------------------------------
# Power-claim formulas (Exercise 6.1, page 118)
# ---------------------------------------------------------------------------

def power_claim_value(t: float, S: float, sigma: float, r: float, T: float, p: float) -> float:
    """Exact price of the power claim phi(S_T) = S_T^p.

    V_t = S_t^p * exp((sigma^2*p/2 + r)*(p-1)*(T-t))

    Derived via risk-neutral expectation under the lognormal distribution.
    Reference: Appendix A.6, page 202.
    """
    tau = T - t
    if tau <= 0.0:
        return S**p
    exponent = (sigma**2 * p / 2.0 + r) * (p - 1.0) * tau
    return S**p * math.exp(exponent)


def power_claim_delta(t: float, S: float, sigma: float, r: float, T: float, p: float) -> float:
    """Delta of the power claim phi(S_T) = S_T^p.

    Delta_t = p * S_t^(p-1) * exp((sigma^2*p/2 + r)*(p-1)*(T-t))

    Reference: Appendix A.6, page 202.
    """
    tau = T - t
    if tau <= 0.0:
        return p * S**(p - 1.0)
    exponent = (sigma**2 * p / 2.0 + r) * (p - 1.0) * tau
    return p * S**(p - 1.0) * math.exp(exponent)


# ---------------------------------------------------------------------------
# Vectorised wrappers for use in Monte Carlo
# ---------------------------------------------------------------------------

def bs_call_vec(
    t: float,
    S: np.ndarray,
    sigma: float,
    T: float,
    K: float,
    r: float,
) -> np.ndarray:
    """Vectorised Black-Scholes call price (r=0 branch kept for efficiency)."""
    tau = T - t
    if tau <= 0.0:
        return np.maximum(S - K, 0.0)
    log_ratio = np.log(S / K)
    dp = (log_ratio + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    dm = dp - sigma * math.sqrt(tau)
    return S * norm.cdf(dp) - K * math.exp(-r * tau) * norm.cdf(dm)


def bs_delta_call_vec(
    t: float,
    S: np.ndarray,
    sigma: float,
    T: float,
    K: float,
    r: float,
) -> np.ndarray:
    """Vectorised Delta of a European call."""
    tau = T - t
    if tau <= 0.0:
        return (S > K).astype(float)
    dp = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    return norm.cdf(dp)


def bs_gamma_vec(
    t: float,
    S: np.ndarray,
    sigma: float,
    T: float,
    K: float,
    r: float,
) -> np.ndarray:
    """Vectorised Gamma of a European option."""
    tau = T - t
    if tau <= 0.0:
        return np.zeros_like(S)
    dp = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    return norm.pdf(dp) / (S * sigma * math.sqrt(tau))
