"""Dupire local-volatility surface recovery.

Implements Eq. 9.7 from Chapter 9:
  f^2(T, K) = partial_T C / (0.5 * K^2 * partial_K^2 C)

Two routes:
  Route A – finite-difference approximations (Eqs. 9.8–9.9)
  Route B – smooth SVI interpolation then analytic differentiation

Also implements the Exercise 9.1 special case: when implied vol depends only
on T, the recovered local vol should be independent of K and satisfy
  sigma^2(T) = (1/(T-t)) * integral_t^T f^2(u, K) du
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from src.black_scholes import bs_call


# ---------------------------------------------------------------------------
# Dupire's formula (Eq. 9.7)
# ---------------------------------------------------------------------------

def dupire_local_vol(
    dC_dT: float,
    dC_dK2: float,
    K: float,
) -> float | None:
    """Recover local variance from Dupire's formula (Eq. 9.7).

    Returns None when the denominator is non-positive (ill-conditioned node).
    """
    denom = 0.5 * K**2 * dC_dK2
    if denom <= 0.0:
        return None
    return dC_dT / denom


# ---------------------------------------------------------------------------
# Route A: finite-difference implementation (Section 9.1, Eqs. 9.8–9.9)
# ---------------------------------------------------------------------------

def dupire_fd(
    C: np.ndarray,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
) -> np.ndarray:
    """Recover local-variance surface f^2 via central finite differences.

    Parameters
    ----------
    C       : call-price matrix, shape (n_T, n_K); C[i, j] = C(T_grid[i], K_grid[j])
    T_grid  : maturity grid, shape (n_T,)
    K_grid  : strike grid, shape (n_K,)

    Returns
    -------
    f2 : local-variance surface, shape (n_T-2, n_K-2), on the interior nodes.
         NaN where Dupire's denominator is non-positive.
    """
    n_T, n_K = C.shape
    assert n_T >= 3 and n_K >= 3, "Need at least 3 maturities and 3 strikes."
    f2 = np.full((n_T - 2, n_K - 2), np.nan)
    for i in range(1, n_T - 1):
        dT = T_grid[i + 1] - T_grid[i - 1]
        for j in range(1, n_K - 1):
            dK = K_grid[j + 1] - K_grid[j - 1]
            # Central difference for partial_T C (Eq. 9.8)
            dC_dT = (C[i + 1, j] - C[i - 1, j]) / dT
            # Central difference for partial_K^2 C (Eq. 9.9)
            dK_step = K_grid[j + 1] - K_grid[j]   # assume uniform; fallback
            dK_uniform = (K_grid[j + 1] - K_grid[j - 1]) / 2.0
            dC_dK2 = (C[i, j + 1] - 2 * C[i, j] + C[i, j - 1]) / dK_uniform**2
            lv = dupire_local_vol(dC_dT, dC_dK2, K_grid[j])
            f2[i - 1, j - 1] = lv if lv is not None else np.nan
    return f2


# ---------------------------------------------------------------------------
# SVI parameterisation (Section 9.1, page 142)
# ---------------------------------------------------------------------------

def svi_implied_vol(
    k: float | np.ndarray,
    x: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    xi: float,
) -> float | np.ndarray:
    """SVI implied-volatility parameterisation (Section 9.1, page 142).

    sigma_SVI(T, k) = sqrt(a + b*(rho*(k-x-m) + sqrt((k-x-m)^2 + xi^2)))
    where k = log K, x = log S0.
    """
    z = k - x - m
    inner = a + b * (rho * z + np.sqrt(z**2 + xi**2))
    return np.sqrt(np.maximum(inner, 0.0))


def fit_svi(
    log_K: np.ndarray,
    x: float,
    sigma_obs: np.ndarray,
    *,
    p0: np.ndarray | None = None,
) -> np.ndarray:
    """Fit SVI parameters to observed implied volatilities via least squares.

    Parameters
    ----------
    log_K     : log-strike values, shape (n_K,)
    x         : log spot, scalar
    sigma_obs : observed implied vols, shape (n_K,)
    p0        : initial guess [a, b, rho, m, xi]; defaults if None

    Returns
    -------
    params : array [a, b, rho, m, xi]
    """
    if p0 is None:
        atm_var = float(np.mean(sigma_obs**2))
        p0 = np.array([atm_var * 0.8, 0.2, -0.3, 0.0, 0.2])

    def residuals(params: np.ndarray) -> np.ndarray:
        a, b, rho, m, xi = params
        # Enforce positivity
        a = abs(a)
        b = max(b, 1e-6)
        xi = max(xi, 1e-6)
        rho = np.clip(rho, -0.999, 0.999)
        model = svi_implied_vol(log_K, x, a, b, rho, m, xi)
        return model - sigma_obs

    result = least_squares(
        residuals,
        p0,
        method="lm",
        ftol=1e-10,
        xtol=1e-10,
    )
    return result.x


# ---------------------------------------------------------------------------
# Route B: SVI-smoothed call surface + Dupire differentiation (Section 9.1)
# ---------------------------------------------------------------------------

def build_svi_call_surface(
    S0: float,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    svi_params_per_T: list[np.ndarray],
    r: float = 0.0,
) -> np.ndarray:
    """Build call-price surface from per-maturity SVI fits.

    C[i, j] = C^BS(0, S0; sigma_SVI(T_i, log K_j), T_i, K_j)
    """
    x = math.log(S0)
    n_T = len(T_grid)
    n_K = len(K_grid)
    C = np.empty((n_T, n_K))
    for i, T in enumerate(T_grid):
        a, b, rho, m, xi = svi_params_per_T[i]
        for j, K in enumerate(K_grid):
            k = math.log(K)
            sig = float(svi_implied_vol(k, x, a, b, rho, m, xi))
            C[i, j] = bs_call(0.0, S0, sig, T, K, r)
    return C


def dupire_svi(
    S0: float,
    T_grid: np.ndarray,
    K_dense: np.ndarray,
    svi_params_per_T: list[np.ndarray],
    r: float = 0.0,
) -> np.ndarray:
    """Route B: SVI-smoothed Dupire surface on interior maturity nodes.

    Computes partial_T C by central difference (Eq. 9.8) and partial_K^2 C
    by differentiating the BS composition analytically (Section 9.1).

    Returns f^2 array of shape (n_T-2, n_K), on the interior maturities and
    all strike nodes.
    """
    x = math.log(S0)
    n_T = len(T_grid)
    n_K = len(K_dense)
    C = build_svi_call_surface(S0, T_grid, K_dense, svi_params_per_T, r)
    f2 = np.full((n_T - 2, n_K), np.nan)
    for i in range(1, n_T - 1):
        dT = T_grid[i + 1] - T_grid[i - 1]
        for j, K in enumerate(K_dense):
            # Approximate partial_T C
            dC_dT = (C[i + 1, j] - C[i - 1, j]) / dT
            # Approximate partial_K^2 C using finite differences on dense grid
            if j == 0 or j == n_K - 1:
                continue
            dK = K_dense[j + 1] - K_dense[j - 1]
            dC_dK2 = (C[i, j + 1] - 2 * C[i, j] + C[i, j - 1]) / (dK / 2.0)**2
            lv = dupire_local_vol(dC_dT, dC_dK2, K)
            f2[i - 1, j] = lv if lv is not None else np.nan
    return f2


# ---------------------------------------------------------------------------
# Exercise 9.1 special case: implied vol = sigma(T) only
# ---------------------------------------------------------------------------

def dupire_term_structure(
    S0: float,
    T_grid: np.ndarray,
    sigma_T: np.ndarray,
    K: float,
    r: float = 0.0,
) -> np.ndarray:
    """Recover f^2(T, K) when implied vol depends only on T.

    Since sigma depends only on T, C(T, K) = C^BS(0, S0; sigma(T), T, K).
    partial_K^2 C is computed via the BS formula and partial_T C via central diffs.

    Returns f^2 array of shape (n_T-2,), on interior maturities.
    """
    n_T = len(T_grid)
    C = np.array([bs_call(0.0, S0, sigma_T[i], T_grid[i], K, r) for i in range(n_T)])
    f2 = np.empty(n_T - 2)
    for i in range(1, n_T - 1):
        dT = T_grid[i + 1] - T_grid[i - 1]
        dC_dT = (C[i + 1] - C[i - 1]) / dT
        # Compute partial_K^2 C numerically
        eps = K * 1e-4
        dC_dK2 = (
            bs_call(0.0, S0, sigma_T[i], T_grid[i], K + eps, r)
            - 2 * C[i]
            + bs_call(0.0, S0, sigma_T[i], T_grid[i], K - eps, r)
        ) / eps**2
        lv = dupire_local_vol(dC_dT, dC_dK2, K)
        f2[i - 1] = lv if lv is not None else np.nan
    return f2


def verify_integral_identity(
    T_grid: np.ndarray,
    f2_interior: np.ndarray,
    sigma_T: np.ndarray,
    t: float = 0.0,
) -> np.ndarray:
    """Verify sigma^2(T) = (1/(T-t)) * integral_t^T f^2(u, K) du.

    Uses trapezoidal integration on interior maturities.
    Returns array of |sigma^2(T_i) - (1/(T_i-t)) * integral|
    for interior indices i=1..n_T-2.
    """
    from scipy.integrate import trapezoid as trap

    n_int = len(f2_interior)  # n_T - 2
    errors = np.empty(n_int)
    interior_T = T_grid[1:-1]
    for idx in range(n_int):
        T_i = interior_T[idx]
        # Integrate f^2 from t to T_i using the available interior points
        sub_T = interior_T[:idx + 1]
        sub_f2 = f2_interior[:idx + 1]
        if len(sub_T) < 2:
            errors[idx] = np.nan
            continue
        integral_val = float(trap(sub_f2, sub_T))
        lhs = sigma_T[1 + idx]**2  # sigma^2(T_{1+idx})
        rhs = integral_val / (T_i - t)
        errors[idx] = abs(lhs - rhs)
    return errors
