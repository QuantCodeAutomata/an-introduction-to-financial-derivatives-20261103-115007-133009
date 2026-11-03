"""Variance-swap replication and volatility-mismatch delta-hedging.

Implements two experiments from Chapter 8 of the notes:

Part A (Section 8.1): P&L of a delta-hedged short call under volatility mismatch.
  P&L_T = 0.5 * integral_0^T Gamma^BS(t, S_t; sigma_2) * (sigma_2^2 - sigma_1^2) * S_t^2 dt

Part B (Section 8.2): Variance-swap floating-leg replication and fair strike.
  integral_0^T sigma_t^2 dt = integral_0^T (2/S_t) dS_t - 2 * log(S_T / S_0)
  Fair strike K = -2 * E_tilde[log(S_T / S_0)]
  Static replication via K = integral_0^S0 (2/K^2) P(T,K) dK + integral_S0^inf (2/K^2) C(T,K) dK
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid

from src.black_scholes import (
    bs_call,
    bs_call_vec,
    bs_delta_call_vec,
    bs_gamma_vec,
    bs_put,
)


# ---------------------------------------------------------------------------
# Part A: Vol-mismatch delta-hedging (Section 8.1)
# ---------------------------------------------------------------------------

def simulate_vol_mismatch(
    S0: float,
    K: float,
    T: float,
    sigma1: float,
    sigma2: float,
    mu: float,
    N: int,
    M: int,
    *,
    r: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate the P&L of a delta-hedged short call under volatility mismatch.

    The true stock dynamics use sigma1; the banker prices and hedges with sigma2.
    With r=0, the portfolio update is:
      X_{t+dt} = X_t + Delta_t * (S_{t+dt} - S_t)   (Section 8.1, page 133)

    Parameters
    ----------
    S0, K, T  : initial price, strike, maturity
    sigma1    : true volatility
    sigma2    : pricing/hedging volatility (sigma2 > sigma1 for positive P&L)
    mu        : physical drift
    N         : number of hedging steps
    M         : number of Monte Carlo paths
    r         : risk-free rate (0 for the section's analysis)
    rng       : optional random Generator

    Returns
    -------
    dict with:
      errors          : array (M,) of X_T - (S_T - K)+
      pnl_identity    : array (M,) of approximate continuous P&L integral
      one_path_S      : first simulated stock path, shape (N+1,)
      one_path_X      : replication portfolio for first path, shape (N+1,)
      one_path_C      : BS option value for first path, shape (N+1,)
      t_grid          : time grid, shape (N+1,)
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = T / N
    t_grid = np.linspace(0.0, T, N + 1)

    # Simulate all paths with true sigma1 (lognormal exact stepping)
    Z = rng.standard_normal((M, N))
    log_increments = (mu - 0.5 * sigma1**2) * dt + sigma1 * math.sqrt(dt) * Z
    log_S = np.empty((M, N + 1))
    log_S[:, 0] = math.log(S0)
    log_S[:, 1:] = math.log(S0) + np.cumsum(log_increments, axis=1)
    S_paths = np.exp(log_S)

    # Initial option value priced at sigma2
    X0 = bs_call(0.0, S0, sigma2, T, K, r)

    errors = np.empty(M)
    pnl_identity = np.empty(M)

    for m in range(M):
        S = S_paths[m]
        X = X0
        pnl_int = 0.0
        for n in range(N):
            t_n = t_grid[n]
            D = float(bs_delta_call_vec(t_n, np.array([S[n]]), sigma2, T, K, r)[0])
            G = float(bs_gamma_vec(t_n, np.array([S[n]]), sigma2, T, K, r)[0])
            dS = S[n + 1] - S[n]
            # Portfolio update with r=0 (page 133): dX = Delta * dS
            X = X + D * dS
            # Approximate continuous P&L identity (Eq. 8 in Section 8.1)
            pnl_int += 0.5 * G * (sigma2**2 - sigma1**2) * S[n]**2 * dt
        errors[m] = X - max(S[-1] - K, 0.0)
        pnl_identity[m] = pnl_int

    # First path details for plotting
    S0_path = S_paths[0]
    X_p = X0
    X_path_single = [X0]
    C_path_single = [bs_call(0.0, S0, sigma2, T, K, r)]
    for n in range(N):
        D = float(bs_delta_call_vec(t_grid[n], np.array([S0_path[n]]), sigma2, T, K, r)[0])
        X_p = X_p + D * (S0_path[n + 1] - S0_path[n])
        X_path_single.append(X_p)
        C_path_single.append(bs_call(t_grid[n + 1], S0_path[n + 1], sigma2, T, K, r))

    return dict(
        errors=errors,
        pnl_identity=pnl_identity,
        one_path_S=S0_path,
        one_path_X=np.array(X_path_single),
        one_path_C=np.array(C_path_single),
        t_grid=t_grid,
    )


# ---------------------------------------------------------------------------
# Part B: Variance-swap replication and fair strike (Section 8.2)
# ---------------------------------------------------------------------------

def simulate_varswap_floating_leg(
    S0: float,
    sigma_t_func,
    T: float,
    N: int,
    M: int,
    *,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate the variance-swap floating leg and replication P&L.

    Under the risk-neutral measure with r=0:
      dS_t = sigma_t * S_t * dW_tilde_t
    Floating leg = integral_0^T sigma_t^2 dt
    Replication: buy log contract payoff -2*log(S_T/S_0), hold Delta_t=2/S_t shares.

    Parameters
    ----------
    S0          : initial stock price
    sigma_t_func: callable(t, S_t) → sigma_t; for constant vol use lambda t, S: sigma
    T           : maturity
    N           : number of time steps
    M           : number of paths
    rng         : optional random Generator

    Returns
    -------
    dict with: floating_leg (M,), replication_pnl (M,), realized_variances (M,)
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = T / N
    t_grid = np.linspace(0.0, T, N + 1)

    floating_legs = np.empty(M)
    replication_pnl = np.empty(M)

    for m in range(M):
        S = S0
        log_contract_value = 0.0  # running stochastic integral 2/S_t * dS_t
        integrated_var = 0.0
        for n in range(N):
            t_n = t_grid[n]
            sig_t = sigma_t_func(t_n, S)
            # Exact lognormal step under RN (r=0): S_{t+dt} = S_t * exp(-0.5*sig^2*dt + sig*sqrt(dt)*Z)
            Z = float(rng.standard_normal())
            dW = math.sqrt(dt) * Z
            dS = sig_t * S * dW  # approximate; exact below
            S_new = S * math.exp(-0.5 * sig_t**2 * dt + sig_t * dW)
            dS_exact = S_new - S
            log_contract_value += (2.0 / S) * dS_exact
            integrated_var += sig_t**2 * dt
            S = S_new

        floating_legs[m] = integrated_var
        # Replication: stoch integral + log contract
        replication_pnl[m] = log_contract_value - 2.0 * math.log(S / S0)

    return dict(
        floating_leg=floating_legs,
        replication_pnl=replication_pnl,
    )


def varswap_fair_strike_static(
    S0: float,
    T: float,
    sigma_func,
    *,
    K_min_frac: float = 0.1,
    K_max_frac: float = 5.0,
    n_strikes: int = 500,
    r: float = 0.0,
) -> float:
    """Compute variance-swap fair strike via the static option-replication formula.

    K = integral_0^{S0} (2/K^2) P_0(T,K) dK + integral_{S0}^inf (2/K^2) C_0(T,K) dK
    (Section 8.2, page 136)

    Parameters
    ----------
    S0         : current stock price
    T          : maturity
    sigma_func : callable(K) → implied volatility at strike K (e.g. const. vol)
    K_min_frac, K_max_frac : range [K_min_frac*S0, K_max_frac*S0] for integration
    n_strikes  : number of quadrature points
    r          : risk-free rate (0 in the notes' analysis)

    Returns
    -------
    Fair variance-swap strike (float).
    """
    K_min = K_min_frac * S0
    K_max = K_max_frac * S0

    # Put integral from K_min to S0
    K_put = np.linspace(K_min, S0, n_strikes // 2 + 1)
    w_put = np.array([bs_put(0.0, S0, sigma_func(k), T, k, r) for k in K_put])
    integrand_put = 2.0 / K_put**2 * w_put
    I_put = trapezoid(integrand_put, K_put)

    # Call integral from S0 to K_max
    K_call = np.linspace(S0, K_max, n_strikes // 2 + 1)
    w_call = np.array([bs_call(0.0, S0, sigma_func(k), T, k, r) for k in K_call])
    integrand_call = 2.0 / K_call**2 * w_call
    I_call = trapezoid(integrand_call, K_call)

    return I_put + I_call


def varswap_fair_strike_mc(
    S0: float,
    sigma_const: float,
    T: float,
    N: int = 1000,
    M: int = 10000,
    *,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate fair strike K = -2 * E_tilde[log(S_T/S_0)] by Monte Carlo.

    Under constant vol with r=0, K = sigma^2 * T (consistency check).
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = T / N
    Z = rng.standard_normal((M, N))
    log_inc = -0.5 * sigma_const**2 * dt + sigma_const * math.sqrt(dt) * Z
    log_ST_S0 = np.sum(log_inc, axis=1)
    return float(-2.0 * np.mean(log_ST_S0))
