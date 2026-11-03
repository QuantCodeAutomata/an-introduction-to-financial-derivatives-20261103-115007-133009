"""Monte Carlo simulation of GBM stock and bond paths.

Simulates log-processes as recommended in Chapter 5.6 (page 106) to reduce
discretisation error:
  Z_t = log S_t → Z_{(n+1)dt} = Z_{n*dt} + (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z^N(0,1)
  Y_t = log B_t → B_t = B0 * exp(r*t)  (exact)
"""

from __future__ import annotations

import numpy as np

from src.black_scholes import power_claim_delta, power_claim_value


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    M: int,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate M paths of a GBM stock under the physical measure.

    Uses exact lognormal stepping (Chapter 5.6, page 106).

    Parameters
    ----------
    S0, mu, sigma : initial price, drift, volatility
    T, N          : maturity and number of time steps
    M             : number of paths
    rng           : optional numpy random Generator for reproducibility

    Returns
    -------
    S_paths : array of shape (M, N+1) with S_paths[:, 0] = S0
    t_grid  : array of length N+1, from 0 to T
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = T / N
    t_grid = np.linspace(0.0, T, N + 1)
    Z = rng.standard_normal((M, N))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.empty((M, N + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum(log_increments, axis=1)
    return np.exp(log_S), t_grid


def simulate_bond_path(B0: float, r: float, t_grid: np.ndarray) -> np.ndarray:
    """Exact bond path B_t = B0 * exp(r*t) (deterministic)."""
    return B0 * np.exp(r * t_grid)


def replicate_power_claim(
    S_path: np.ndarray,
    B_path: np.ndarray,
    t_grid: np.ndarray,
    sigma: float,
    r: float,
    T: float,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the discrete self-financing replication portfolio for one path.

    Implements the update rule (Appendix A.6, page 202):
      X_{t+dt} = X_t + Delta_t*(S_{t+dt} - S_t) + (X_t - Delta_t*S_t)*(B_{t+dt} - B_t)/B_t

    Parameters
    ----------
    S_path : stock price path, shape (N+1,)
    B_path : bond path, shape (N+1,)
    t_grid : time grid, shape (N+1,)
    sigma, r, T, p : model and payoff parameters

    Returns
    -------
    V_path : exact option value along the path, shape (N+1,)
    X_path : discrete replication portfolio value along the path, shape (N+1,)
    """
    N = len(t_grid) - 1
    V_path = np.array([power_claim_value(t_grid[n], S_path[n], sigma, r, T, p) for n in range(N + 1)])
    X_path = np.empty(N + 1)
    X_path[0] = V_path[0]
    for n in range(N):
        D = power_claim_delta(t_grid[n], S_path[n], sigma, r, T, p)
        dS = S_path[n + 1] - S_path[n]
        dB = B_path[n + 1] - B_path[n]
        X_path[n + 1] = X_path[n] + D * dS + (X_path[n] - D * S_path[n]) * dB / B_path[n]
    return V_path, X_path


def compute_hedging_errors(
    S0: float,
    mu: float,
    sigma: float,
    r: float,
    T: float,
    p: float,
    N: int,
    M: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute terminal hedging errors X_T - V_T over M Monte Carlo paths.

    Returns array of shape (M,).
    """
    if rng is None:
        rng = np.random.default_rng()
    B0 = 1.0
    S_paths, t_grid = simulate_gbm_paths(S0, mu, sigma, T, N, M, rng=rng)
    B_path = simulate_bond_path(B0, r, t_grid)
    errors = np.empty(M)
    for m in range(M):
        V_path, X_path = replicate_power_claim(S_paths[m], B_path, t_grid, sigma, r, T, p)
        errors[m] = X_path[-1] - V_path[-1]
    return errors
