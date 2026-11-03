"""Binomial tree pricing and hedging for European and American options.

Implements the CRR (Cox-Ross-Rubinstein) parameterisation described in
Chapter 3 of the notes: u = exp(sigma*sqrt(dt)), d = exp(-sigma*sqrt(dt)),
disc = exp(-r*dt), p_tilde = (exp(r*dt) - d) / (u - d).

Reference: Chapter 3, Sections 3.1–3.2 and Exercises 3.4–3.5.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Parameter calibration (Section 3.2.4)
# ---------------------------------------------------------------------------

def calibrate_crr(
    sigma: float,
    r: float,
    T: float,
    N: int,
) -> dict[str, float]:
    """Return CRR tree parameters for given volatility, rate, maturity, steps.

    Parameters
    ----------
    sigma : annualised volatility
    r     : continuously compounded risk-free rate
    T     : maturity in years
    N     : number of steps

    Returns
    -------
    dict with keys: dt, u, d, disc, p_tilde, q_tilde
    """
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-sigma * math.sqrt(dt))
    disc = math.exp(-r * dt)
    er = math.exp(r * dt)
    p_tilde = (er - d) / (u - d)
    q_tilde = 1.0 - p_tilde
    if not (0.0 < p_tilde < 1.0):
        raise ValueError(
            f"No-arbitrage violated: p_tilde={p_tilde:.4f}. "
            "Check that d < exp(r*dt) < u."
        )
    return dict(dt=dt, u=u, d=d, disc=disc, p_tilde=p_tilde, q_tilde=q_tilde)


# ---------------------------------------------------------------------------
# One-step pricing (Section 3.1.1 and 3.1.2)
# ---------------------------------------------------------------------------

def price_european_onestep(
    S0: float,
    u: float,
    d: float,
    r: float,
    payoff: Callable[[float], float],
) -> dict[str, float]:
    """Price and hedge a one-step European option by replication and risk-neutral valuation.

    No-arbitrage requires 0 < d < 1+r < u  (discrete-rate convention here).

    Parameters
    ----------
    S0     : initial stock price
    u      : up factor
    d      : down factor
    r      : one-step risk-free rate (simple, not continuously compounded)
    payoff : callable mapping terminal stock price to option payoff

    Returns
    -------
    dict with keys: V0 (price), Delta0 (hedge ratio), p_tilde, q_tilde
    """
    S_up = u * S0
    S_dn = d * S0
    phi_up = payoff(S_up)
    phi_dn = payoff(S_dn)

    # Replication: solve for Delta0 and X0
    # Delta0*S_up + (X0 - Delta0*S0)*(1+r) = phi_up
    # Delta0*S_dn + (X0 - Delta0*S0)*(1+r) = phi_dn
    Delta0 = (phi_up - phi_dn) / (S_up - S_dn)
    X0 = (phi_up - Delta0 * S_up) / (1.0 + r) + Delta0 * S0

    # Risk-neutral valuation
    one_plus_r = 1.0 + r
    p_tilde = (one_plus_r - d) / (u - d)
    q_tilde = 1.0 - p_tilde
    V0_rn = (p_tilde * phi_up + q_tilde * phi_dn) / one_plus_r

    assert abs(X0 - V0_rn) < 1e-10, (
        "Replication price and risk-neutral price disagree: "
        f"{X0:.8f} vs {V0_rn:.8f}"
    )
    return dict(V0=X0, Delta0=Delta0, p_tilde=p_tilde, q_tilde=q_tilde)


# ---------------------------------------------------------------------------
# N-step European pricing (Sections 3.2.1 and 3.2.2)
# ---------------------------------------------------------------------------

def build_stock_lattice(S0: float, u: float, d: float, N: int) -> np.ndarray:
    """Construct the recombining stock price lattice.

    Node (n, i) holds u^i * d^(n-i) * S0, with i indexing the number of
    up-moves in n steps.  Shape: (N+1, N+1), lower-left triangle populated.
    """
    lattice = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        for i in range(n + 1):
            lattice[n, i] = u**i * d**(n - i) * S0
    return lattice


def price_european_binomial(
    S0: float,
    u: float,
    d: float,
    disc: float,
    p_tilde: float,
    q_tilde: float,
    N: int,
    payoff: Callable[[float], float],
    *,
    return_lattice: bool = False,
) -> dict:
    """Price and hedge an N-step European option by backward recursion.

    Parameters correspond to Chapter 3, Eq. 3.7 and 3.10.
    Uses a 1-D array updated in place to keep memory O(N).
    When return_lattice=True, the full value lattice is returned (O(N^2) memory).

    Returns
    -------
    dict with: V0, Delta0, and optionally value_lattice, delta_lattice
    """
    # Terminal payoffs at step N: node i has stock u^i * d^(N-i) * S0
    V = np.array([payoff(u**i * d**(N - i) * S0) for i in range(N + 1)])

    if return_lattice:
        val_lat = np.zeros((N + 1, N + 1))
        del_lat = np.zeros((N + 1, N + 1))
        val_lat[N, :N + 1] = V.copy()

    # Backward induction from step N-1 down to step 1 (exclusive of step 0)
    # Node (n, i) has UP child (n+1, i+1) → V[i+1] and DOWN child (n+1, i) → V[i]
    # Eq. 3.7: X_n^(i) = disc * (p_tilde * X_{n+1}^(i+1) + q_tilde * X_{n+1}^(i))
    # Eq. 3.10: Delta_n^(i) = (X_{n+1}^(i+1) - X_{n+1}^(i)) / (S_{n+1}^(i+1) - S_{n+1}^(i))
    for n in range(N - 1, 0, -1):
        S_up = np.array([u**(i + 1) * d**(n - i) * S0 for i in range(n + 1)])
        S_dn = np.array([u**i * d**(n + 1 - i) * S0 for i in range(n + 1)])
        if return_lattice:
            delta_n = (V[1:n + 2] - V[:n + 1]) / (S_up - S_dn)
            del_lat[n, :n + 1] = delta_n
        V[:n + 1] = disc * (p_tilde * V[1:n + 2] + q_tilde * V[:n + 1])
        if return_lattice:
            val_lat[n, :n + 1] = V[:n + 1]

    # Step 0: V[1] is the UP child (i=1 up-moves), V[0] is the DOWN child (i=0)
    S_up_0 = u * S0
    S_dn_0 = d * S0
    Delta0 = (V[1] - V[0]) / (S_up_0 - S_dn_0)
    V0 = disc * (p_tilde * V[1] + q_tilde * V[0])

    if return_lattice:
        val_lat[0, 0] = V0
        del_lat[0, 0] = Delta0
        return dict(
            V0=V0,
            Delta0=Delta0,
            value_lattice=val_lat,
            delta_lattice=del_lat,
        )
    return dict(V0=V0, Delta0=Delta0)


# ---------------------------------------------------------------------------
# N-step American pricing (Section 3.2.3)
# ---------------------------------------------------------------------------

def price_american_binomial(
    S0: float,
    u: float,
    d: float,
    disc: float,
    p_tilde: float,
    q_tilde: float,
    N: int,
    payoff: Callable[[float], float],
    *,
    return_lattice: bool = False,
) -> dict:
    """Price and hedge an N-step American option by backward recursion.

    At each node the holder chooses the maximum of immediate exercise and
    discounted continuation (Chapter 3, Section 3.2.3).

    Returns
    -------
    dict with: V0, Delta0, and optionally value_lattice, delta_lattice,
    exercise_lattice (True where early exercise is optimal).
    """
    q_tilde_val = 1.0 - p_tilde
    # Terminal payoffs
    V = np.array([payoff(u**i * d**(N - i) * S0) for i in range(N + 1)])

    if return_lattice:
        val_lat = np.zeros((N + 1, N + 1))
        del_lat = np.zeros((N + 1, N + 1))
        ex_lat = np.zeros((N + 1, N + 1), dtype=bool)
        val_lat[N, :N + 1] = V.copy()
        ex_lat[N, :N + 1] = True  # terminal: always exercise if in the money

    # Node (n, i): UP child is (n+1, i+1) → V[i+1], DOWN child is (n+1, i) → V[i]
    for n in range(N - 1, 0, -1):
        S_cur = np.array([u**i * d**(n - i) * S0 for i in range(n + 1)])
        S_up = np.array([u**(i + 1) * d**(n - i) * S0 for i in range(n + 1)])
        S_dn = np.array([u**i * d**(n + 1 - i) * S0 for i in range(n + 1)])
        cont = disc * (p_tilde * V[1:n + 2] + q_tilde_val * V[:n + 1])
        exer = np.array([payoff(S_cur[i]) for i in range(n + 1)])
        V_new = np.maximum(exer, cont)
        if return_lattice:
            del_lat[n, :n + 1] = (V[1:n + 2] - V[:n + 1]) / (S_up - S_dn)
            ex_lat[n, :n + 1] = exer >= cont
            val_lat[n, :n + 1] = V_new
        V[:n + 1] = V_new

    # Step 0: V[1] is UP child, V[0] is DOWN child
    S_up_0 = u * S0
    S_dn_0 = d * S0
    cont0 = disc * (p_tilde * V[1] + q_tilde_val * V[0])
    exer0 = payoff(S0)
    V0 = max(exer0, cont0)
    Delta0 = (V[1] - V[0]) / (S_up_0 - S_dn_0)

    if return_lattice:
        val_lat[0, 0] = V0
        del_lat[0, 0] = Delta0
        ex_lat[0, 0] = exer0 >= cont0
        return dict(
            V0=V0,
            Delta0=Delta0,
            value_lattice=val_lat,
            delta_lattice=del_lat,
            exercise_lattice=ex_lat,
        )
    return dict(V0=V0, Delta0=Delta0)


# ---------------------------------------------------------------------------
# Standard payoffs
# ---------------------------------------------------------------------------

def call_payoff(K: float) -> Callable[[float], float]:
    """Return a European/American call payoff function with strike K."""
    return lambda s: max(s - K, 0.0)


def put_payoff(K: float) -> Callable[[float], float]:
    """Return a European/American put payoff function with strike K."""
    return lambda s: max(K - s, 0.0)
