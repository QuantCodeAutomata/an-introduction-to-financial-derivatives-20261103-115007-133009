"""Heston stochastic-volatility model: pricing and calibration.

The Heston model (Chapter 10, Example 10.0.1, page 145) is the one-factor SV
model with coefficients:
  alpha(y) = kappa*(theta - y),  beta(y) = delta*sqrt(y),  sigma(y) = sqrt(y)

Pricing uses the semi-analytical characteristic-function representation
(Heston 1993) rather than the Riccati ODE formulas, which the notes omit
(page 152: "We will not give the expressions for C and D here").

Calibration minimises the sum of squared differences between Heston and
market implied volatilities (Section 10.4, Eq. on page 153).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize

from src.black_scholes import bs_call
from src.implied_vol import implied_vol_call


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


def _heston_cf(
    u: complex,
    tau: float,
    v0: float,
    kappa: float,
    theta: float,
    delta: float,
    rho: float,
    r: float,
    S: float,
) -> complex:
    """Heston characteristic function phi(u) = E[exp(i*u*log(S_T)) | S_0, v_0].

    Uses the Heston (1993) formulation; see e.g. Heston (1993), Eq. (17)–(18).
    The standard representation with the complex square-root branch chosen to
    avoid discontinuities.
    """
    # Compute d = sqrt((kappa - i*rho*delta*u)^2 + delta^2*(i*u + u^2))
    kappa_adj = kappa - 1j * rho * delta * u
    d = cmath_sqrt(kappa_adj**2 + delta**2 * (1j * u + u**2))
    g = (kappa_adj - d) / (kappa_adj + d)
    exp_d_tau = cmath_exp(-d * tau)
    # Ensure no division by zero
    denom_D = 1.0 - g * exp_d_tau
    if abs(denom_D) < 1e-15:
        denom_D = 1e-15
    denom_C_log = 1.0 - g * exp_d_tau
    denom_C = 1.0 - g
    if abs(denom_C) < 1e-15:
        denom_C = 1e-15
    D = (kappa_adj - d) / delta**2 * (1.0 - exp_d_tau) / denom_D
    C = (
        r * 1j * u * tau
        + kappa * theta / delta**2 * (
            (kappa_adj - d) * tau - 2.0 * cmath_log(denom_C_log / denom_C)
        )
    )
    return cmath_exp(C + D * v0 + 1j * u * math.log(S))


def cmath_sqrt(z: complex) -> complex:
    """Branch cut on the negative real axis (standard Python cmath)."""
    import cmath
    return cmath.sqrt(z)


def cmath_exp(z: complex) -> complex:
    import cmath
    return cmath.exp(z)


def cmath_log(z: complex) -> complex:
    import cmath
    return cmath.log(z)


def heston_call(
    S: float,
    K: float,
    T: float,
    params: HestonParams,
    *,
    t: float = 0.0,
    n_integration: int = 200,
    upper_limit: float = 500.0,
) -> float:
    """Heston European call price via Fourier inversion (Heston 1993).

    Uses the Carr-Madan / Lewis representation:
      C = S * exp(-r*tau) * P1 - K * exp(-r*tau) * P2
    where Pj are recovered by integrating the characteristic function.

    Parameters
    ----------
    S, K, T  : spot, strike, maturity
    params   : HestonParams
    t        : current time (default 0)
    n_integration : Gauss-Laguerre quadrature points
    upper_limit   : integration truncation

    Returns
    -------
    European call price (float).
    """
    tau = T - t
    if tau <= 0.0:
        return max(S - K, 0.0)
    v0 = params.v0
    kappa = params.kappa
    theta = params.theta
    delta = params.delta
    rho = params.rho
    r = params.r
    log_K = math.log(K)

    def integrand_P1(u: float) -> float:
        # P1 uses phi evaluated at u - i
        cf = _heston_cf(
            complex(u, -1.0), tau, v0, kappa, theta, delta, rho, r, S
        )
        # Normalise by phi(-i) to get the stock-measure CF
        cf_norm = _heston_cf(
            complex(0.0, -1.0), tau, v0, kappa, theta, delta, rho, r, S
        )
        val = cf / cf_norm
        re = (val * cmath_exp(-1j * u * log_K) / (1j * u)).real
        return re

    def integrand_P2(u: float) -> float:
        cf = _heston_cf(
            complex(u, 0.0), tau, v0, kappa, theta, delta, rho, r, S
        )
        re = (cf * cmath_exp(-1j * u * log_K) / (1j * u)).real
        return re

    I1, _ = quad(integrand_P1, 1e-6, upper_limit, limit=n_integration)
    I2, _ = quad(integrand_P2, 1e-6, upper_limit, limit=n_integration)

    P1 = 0.5 + I1 / math.pi
    P2 = 0.5 + I2 / math.pi
    call = math.exp(-r * tau) * (S * math.exp(r * tau) * P1 - K * P2)
    return max(call, max(S * math.exp(-r * tau) - K * math.exp(-r * tau), 0.0))


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

    Returns None if conversion fails (price outside no-arb bounds or numerical issue).
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

    Parameters
    ----------
    S             : current spot price
    market_ivols  : matrix of market implied vols, shape (len(T_grid), len(K_grid))
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
    mkt_flat = market_ivols.flatten()

    def objective(x: np.ndarray) -> float:
        v0, kappa, theta, delta, rho = x
        p = HestonParams(v0=v0, kappa=kappa, theta=theta, delta=delta, rho=rho, r=r)
        sq_errors = []
        for i, T in enumerate(T_arr):
            for j, K in enumerate(K_arr):
                iv_h = heston_implied_vol(S, K, T, p, t=t, r=r)
                if iv_h is None:
                    sq_errors.append(1.0)  # large penalty for failed pricing
                else:
                    err = market_ivols[i, j] - iv_h
                    sq_errors.append(err**2)
        return sum(sq_errors)

    # Global optimisation first (DE), then local refinement
    result_de = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=200,
        tol=1e-6,
        popsize=12,
        mutation=(0.5, 1),
        recombination=0.7,
        workers=1,
    )
    result = minimize(
        objective,
        result_de.x,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
    )
    v0, kappa, theta, delta, rho = result.x
    calibrated = HestonParams(v0=v0, kappa=kappa, theta=theta, delta=delta, rho=rho, r=r)
    n_obs = len(T_arr) * len(K_arr)
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
    ivols = np.full((len(T_grid), len(K_grid)), np.nan)
    for i, T in enumerate(T_grid):
        for j, K in enumerate(K_grid):
            iv = heston_implied_vol(S, K, T, params, t=t)
            if iv is not None:
                ivols[i, j] = iv
    return ivols
