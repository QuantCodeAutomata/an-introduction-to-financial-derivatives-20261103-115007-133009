"""Heston stochastic-volatility model: pricing and calibration.

The Heston model (Chapter 10, Example 10.0.1, page 145) is the one-factor SV
model with coefficients:
  alpha(y) = kappa*(theta - y),  beta(y) = delta*sqrt(y),  sigma(y) = sqrt(y)

Pricing uses the semi-analytical characteristic-function representation
(Heston 1993) rather than the Riccati ODE formulas, which the notes omit
(page 152: "We will not give the expressions for C and D here").
Fourier inversion uses vectorised Gauss-Legendre quadrature (Gil-Pelaez 1951).

Calibration minimises the sum of squared differences between Heston and
market implied volatilities (Section 10.4, Eq. on page 153).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from src.implied_vol import implied_vol_call

# Pre-compute 64-point Gauss-Legendre nodes/weights on [0, 1] once at import.
_GL_NODES_01, _GL_WEIGHTS_01 = np.polynomial.legendre.leggauss(64)


@dataclass
class HestonParams:
    """Parameters of the Heston model.

    v0    : initial variance Y_0 = sigma_0^2  (part of Xi in calibration)
    kappa : mean-reversion speed
    theta : long-run variance
    delta : volatility of variance (vol-of-vol)
    rho   : correlation between stock and variance Brownian motions
    r     : risk-free rate (often 0 in Chapter 10's examples)
    """
    v0: float
    kappa: float
    theta: float
    delta: float
    rho: float
    r: float = 0.0


def _heston_cf_vec(
    u: np.ndarray,
    tau: float,
    v0: float,
    kappa: float,
    theta: float,
    delta: float,
    rho: float,
    r: float,
    log_S: float,
) -> np.ndarray:
    """Vectorised Heston log-stock CF phi(u) for a 1-D complex array u.

    Uses the formulation from Heston (1993), Eqs. (17)–(18), with the
    square-root branch chosen to avoid the discontinuity noted by
    Albrecher et al. (2007): d is computed with np.sqrt (branch on negative
    real axis) and the log of the ratio uses np.log with the same branch.
    """
    kappa_adj = kappa - 1j * rho * delta * u          # kappa - i*rho*delta*u
    d = np.sqrt(kappa_adj**2 + delta**2 * (1j * u + u**2))
    g = (kappa_adj - d) / (kappa_adj + d)
    exp_d_tau = np.exp(-d * tau)
    denom_D = 1.0 - g * exp_d_tau
    denom_C = 1.0 - g
    # Guard against exact zeros (negligible numerical event)
    denom_D = np.where(np.abs(denom_D) < 1e-300, 1e-300, denom_D)
    denom_C = np.where(np.abs(denom_C) < 1e-300, 1e-300, denom_C)
    D = (kappa_adj - d) / delta**2 * (1.0 - exp_d_tau) / denom_D
    C = (
        r * 1j * u * tau
        + kappa * theta / delta**2 * (
            (kappa_adj - d) * tau - 2.0 * np.log(denom_D / denom_C)
        )
    )
    return np.exp(C + D * v0 + 1j * u * log_S)


def heston_call(
    S: float,
    K: float,
    T: float,
    params: HestonParams,
    *,
    t: float = 0.0,
    upper_limit: float = 200.0,
) -> float:
    """Heston European call price via Gil-Pelaez Fourier inversion.

    Uses the two-integral representation (Heston 1993):
      C = S * P1 - K * exp(-r*tau) * P2
    with P1, P2 computed by vectorised 64-point Gauss-Legendre quadrature
    on [0, upper_limit].

    Parameters
    ----------
    S, K, T      : spot, strike, maturity
    params       : HestonParams
    t            : current time
    upper_limit  : Fourier truncation (200 is sufficient for typical params)

    Returns
    -------
    European call price (float ≥ intrinsic value).
    """
    tau = T - t
    if tau <= 0.0:
        return max(S - K, 0.0)

    v0, kappa, theta, delta, rho, r = (
        params.v0, params.kappa, params.theta,
        params.delta, params.rho, params.r,
    )
    log_S, log_K = math.log(S), math.log(K)

    # Map GL nodes from [-1,1] to [0, upper_limit]
    u = (_GL_NODES_01 + 1.0) / 2.0 * upper_limit   # shape (64,)
    w = _GL_WEIGHTS_01 / 2.0 * upper_limit           # shape (64,)

    # P2: integrand Re[ phi(u) * exp(-i*u*log K) / (i*u) ]
    phi2 = _heston_cf_vec(u + 0j, tau, v0, kappa, theta, delta, rho, r, log_S)
    integ2 = (phi2 * np.exp(-1j * u * log_K) / (1j * u)).real
    P2 = 0.5 + np.dot(w, integ2) / math.pi

    # P1: integrand Re[ phi(u-i) / phi(-i) * exp(-i*u*log K) / (i*u) ]
    phi1 = _heston_cf_vec(u - 1j, tau, v0, kappa, theta, delta, rho, r, log_S)
    phi_norm = _heston_cf_vec(
        np.array([-1j]), tau, v0, kappa, theta, delta, rho, r, log_S
    )[0]
    integ1 = (phi1 / phi_norm * np.exp(-1j * u * log_K) / (1j * u)).real
    P1 = 0.5 + np.dot(w, integ1) / math.pi

    call = math.exp(-r * tau) * (S * math.exp(r * tau) * P1 - K * P2)
    intrinsic = max(S * math.exp(-r * tau) - K * math.exp(-r * tau), 0.0)
    return max(float(np.real(call)), intrinsic)


def _heston_calls_for_params(
    S: float,
    T_arr: np.ndarray,
    K_arr: np.ndarray,
    v0: float,
    kappa: float,
    theta: float,
    delta: float,
    rho: float,
    r: float,
    t: float,
    upper_limit: float = 200.0,
) -> np.ndarray:
    """Vectorised call prices for a (T, K) grid, reusing one CF per maturity."""
    log_S = math.log(S)
    u = (_GL_NODES_01 + 1.0) / 2.0 * upper_limit
    w = _GL_WEIGHTS_01 / 2.0 * upper_limit
    prices = np.empty((len(T_arr), len(K_arr)))
    for i, T in enumerate(T_arr):
        tau = T - t
        if tau <= 0.0:
            for j, K in enumerate(K_arr):
                prices[i, j] = max(S - K, 0.0)
            continue
        phi2 = _heston_cf_vec(u + 0j, tau, v0, kappa, theta, delta, rho, r, log_S)
        phi1 = _heston_cf_vec(u - 1j, tau, v0, kappa, theta, delta, rho, r, log_S)
        phi_norm = _heston_cf_vec(
            np.array([-1j]), tau, v0, kappa, theta, delta, rho, r, log_S
        )[0]
        for j, K in enumerate(K_arr):
            log_K = math.log(K)
            integ2 = (phi2 * np.exp(-1j * u * log_K) / (1j * u)).real
            P2 = 0.5 + np.dot(w, integ2) / math.pi
            integ1 = (phi1 / phi_norm * np.exp(-1j * u * log_K) / (1j * u)).real
            P1 = 0.5 + np.dot(w, integ1) / math.pi
            call = math.exp(-r * tau) * (S * math.exp(r * tau) * P1 - K * P2)
            intrinsic = max(S * math.exp(-r * tau) - K * math.exp(-r * tau), 0.0)
            prices[i, j] = max(float(np.real(call)), intrinsic)
    return prices


def heston_implied_vol(
    S: float,
    K: float,
    T: float,
    params: HestonParams,
    *,
    t: float = 0.0,
    r: float | None = None,
) -> float | None:
    """Convert Heston call price to BS implied volatility.

    Returns None if conversion fails (price outside no-arb bounds).
    """
    r_use = r if r is not None else params.r
    price = heston_call(S, K, T, params, t=t)
    try:
        return implied_vol_call(price, t, S, T, K, r_use)
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Calibration (Section 10.4, page 153)
# ---------------------------------------------------------------------------

def calibrate_heston(
    S: float,
    market_ivols: np.ndarray,
    T_grid: Sequence[float],
    K_grid: Sequence[float],
    *,
    r: float = 0.0,
    t: float = 0.0,
    bounds: list[tuple[float, float]] | None = None,
    seed: int = 42,
) -> tuple[HestonParams, float]:
    """Calibrate Heston parameters to market implied volatilities.

    Minimises: sum_{(T,K)} (sigma^M(T,K) - sigma^H(T,K; Xi))^2  (Section 10.4)

    Uses differential evolution (global) followed by Nelder-Mead (local).
    Pricing uses vectorised GL quadrature for speed.

    Parameters
    ----------
    S             : current spot price
    market_ivols  : matrix of market implied vols, shape (nT, nK)
    T_grid        : list of maturities
    K_grid        : list of strikes
    r             : risk-free rate
    t             : current time
    bounds        : list of (lo, hi) for [v0, kappa, theta, delta, rho]
    seed          : random seed for differential evolution

    Returns
    -------
    (calibrated_params, rmse) where rmse is root-mean-square error in vol.
    """
    if bounds is None:
        bounds = [
            (1e-4, 0.5),    # v0
            (0.1, 10.0),    # kappa
            (1e-4, 0.5),    # theta
            (1e-4, 1.5),    # delta
            (-0.99, 0.99),  # rho
        ]

    T_arr = np.array(T_grid)
    K_arr = np.array(K_grid)
    n_obs = len(T_arr) * len(K_arr)
    mkt_flat = market_ivols.flatten()

    from src.black_scholes import bs_call  # local import to avoid circularity

    def _ivols_from_prices(prices: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Batch implied-vol inversion; returns NaN for out-of-bounds nodes."""
        v0_, kappa_, theta_, delta_, rho_ = x
        ivols = np.full(prices.shape, np.nan)
        for i, T in enumerate(T_arr):
            tau = T - t
            for j, K in enumerate(K_arr):
                try:
                    iv = implied_vol_call(float(prices[i, j]), t, S, T, K, r)
                    ivols[i, j] = iv
                except (ValueError, RuntimeError):
                    pass
        return ivols

    def objective(x: np.ndarray) -> float:
        v0_, kappa_, theta_, delta_, rho_ = x
        prices = _heston_calls_for_params(
            S, T_arr, K_arr, v0_, kappa_, theta_, delta_, rho_, r, t
        )
        ivols = _ivols_from_prices(prices, x)
        mask = ~np.isnan(ivols.flatten())
        if mask.sum() == 0:
            return float(n_obs)  # maximum penalty
        err = mkt_flat[mask] - ivols.flatten()[mask]
        penalty = float(n_obs - mask.sum())  # penalty for failed nodes
        return float(np.sum(err**2)) + penalty

    result_de = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=80,
        tol=1e-5,
        popsize=8,
        mutation=(0.5, 1),
        recombination=0.7,
        workers=1,
    )
    result = minimize(
        objective,
        result_de.x,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-7},
    )
    v0, kappa, theta, delta, rho = result.x
    calibrated = HestonParams(v0=v0, kappa=kappa, theta=theta, delta=delta, rho=rho, r=r)
    rmse = math.sqrt(result.fun / n_obs)
    return calibrated, rmse


def heston_surface(
    S: float,
    T_grid: Sequence[float],
    K_grid: Sequence[float],
    params: HestonParams,
    *,
    t: float = 0.0,
) -> np.ndarray:
    """Compute Heston implied-volatility surface for a T×K grid.

    Returns matrix of shape (len(T_grid), len(K_grid)) with NaN for failed nodes.
    """
    T_arr = np.array(T_grid)
    K_arr = np.array(K_grid)
    prices = _heston_calls_for_params(
        S, T_arr, K_arr,
        params.v0, params.kappa, params.theta, params.delta, params.rho,
        params.r, t,
    )
    ivols = np.full((len(T_grid), len(K_grid)), np.nan)
    for i, T in enumerate(T_arr):
        for j, K in enumerate(K_arr):
            try:
                iv = implied_vol_call(float(prices[i, j]), t, S, T, K, params.r)
                ivols[i, j] = iv
            except (ValueError, RuntimeError):
                pass
    return ivols
