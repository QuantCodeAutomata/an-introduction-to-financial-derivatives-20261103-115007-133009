"""Experiment parameters for all five experiments.

All values are module-level constants.  Each variable has a comment stating
its meaning, unit, and source (paper section/equation or "not specified in
the paper").
"""

# ---------------------------------------------------------------------------
# Exp 1 – Binomial model
# ---------------------------------------------------------------------------

# One-step European call example (Chapter 3, Example 3.1.2, page 53)
BINOM_ONESTEP_S0: float = 100.0   # initial stock price, $
BINOM_ONESTEP_U: float = 1.1      # up factor, dimensionless
BINOM_ONESTEP_D: float = 0.9      # down factor, dimensionless
BINOM_ONESTEP_R: float = 0.0      # risk-free rate per step, fraction
BINOM_ONESTEP_K: float = 100.0    # strike, $
# Expected call V0 = 5 (page 53)

# Two-step European call example (Chapter 3, Section 3.2.1, page 60)
BINOM_TWOSTEP_S0: float = 100.0
BINOM_TWOSTEP_U: float = 1.1
BINOM_TWOSTEP_D: float = 0.9
BINOM_TWOSTEP_R: float = 0.0
BINOM_TWOSTEP_K: float = 100.0
BINOM_TWOSTEP_N: int = 2
# Expected call V0 = 5.25 (page 60)

# European put – CRR calibrated tree (Exercise 3.4, page 69; Appendix A, page 178)
BINOM_EUR_S0: float = 50.0        # initial stock price, $
BINOM_EUR_T: float = 2.0          # maturity, years
BINOM_EUR_K: float = 55.0         # strike, $
BINOM_EUR_R: float = 0.05         # risk-free rate, annual continuously compounded
BINOM_EUR_SIGMA: float = 0.10     # volatility, annual
BINOM_EUR_N: int = 50             # number of binomial steps
# Expected: V0 ≈ 2.6955, Delta0 ≈ -0.4623 (page 178)

# American put – same parameters as European put for comparison (Exercise 3.5, page 70)
BINOM_AM_S0: float = 50.0
BINOM_AM_T: float = 2.0
BINOM_AM_K: float = 55.0
BINOM_AM_R: float = 0.05
BINOM_AM_SIGMA: float = 0.10
BINOM_AM_N: int = 50

# ---------------------------------------------------------------------------
# Exp 2 – Black-Scholes
# ---------------------------------------------------------------------------

# Power option Monte Carlo simulation (Exercise 6.1, page 118-119)
BS_S0: float = 1.0        # initial stock price
BS_B0: float = 1.0        # initial bond price
BS_MU: float = 0.1        # physical drift, annual
BS_SIGMA: float = 0.2     # volatility, annual
BS_R: float = 0.03        # risk-free rate, annual continuously compounded
BS_T: float = 1.0         # maturity, years
BS_P: float = -2.0        # power-claim exponent (payoff = S_T^p)
BS_N: int = 100           # number of time steps for discrete hedging
BS_M: int = 1000          # number of Monte Carlo paths

# Black-Scholes vanilla option example for put-call parity check
# (not specified explicitly in the paper; standard ATM example for validation)
BS_VANILLA_S0: float = 100.0
BS_VANILLA_K: float = 100.0
BS_VANILLA_SIGMA: float = 0.2
BS_VANILLA_R: float = 0.05
BS_VANILLA_T: float = 1.0

# ---------------------------------------------------------------------------
# Exp 3 – Stochastic volatility / variance swap
# ---------------------------------------------------------------------------

# Vol-mismatch delta-hedging experiment (Exercise 8.1, page 137)
VOL_MISMATCH_S0: float = 100.0   # initial stock price
VOL_MISMATCH_K: float = 100.0    # strike
VOL_MISMATCH_T: float = 1.0      # maturity, years
VOL_MISMATCH_SIGMA1: float = 0.15  # true volatility
VOL_MISMATCH_SIGMA2: float = 0.20  # pricing/hedging volatility (sigma2 > sigma1)
VOL_MISMATCH_MU: float = 0.05    # physical drift
VOL_MISMATCH_N: int = 100        # hedging steps; not specified in paper
VOL_MISMATCH_M: int = 1000       # Monte Carlo paths; not specified in paper

# Variance-swap static replication (Chapter 8, Section 8.2, pages 135-136)
VARSWAP_S0: float = 100.0        # initial stock price
VARSWAP_SIGMA: float = 0.20      # constant vol for BS benchmark (not specified)
VARSWAP_T: float = 1.0           # maturity, years
# Fair strike in constant-vol BS world with r=0 equals sigma^2*T (page 136, consistency check)
VARSWAP_N_STRIKES: int = 200     # strike grid points; not specified in paper
VARSWAP_K_MIN_FRAC: float = 0.2  # min strike as fraction of S0; not specified
VARSWAP_K_MAX_FRAC: float = 3.0  # max strike as fraction of S0; not specified

# ---------------------------------------------------------------------------
# Exp 4 – Dupire local volatility
# ---------------------------------------------------------------------------

# Synthetic implied-vol surface for Dupire calibration
DUPIRE_S0: float = 100.0    # spot, $
# r=0 assumed throughout Chapter 9 (page 137)
DUPIRE_R: float = 0.0
# Maturity grid: not specified in paper; representative values
DUPIRE_T_MIN: float = 0.25   # min maturity, years; not specified in paper
DUPIRE_T_MAX: float = 2.0    # max maturity, years; not specified in paper
DUPIRE_N_T: int = 8          # number of maturities; not specified in paper
DUPIRE_K_MIN_FRAC: float = 0.6   # min strike / S0; not specified
DUPIRE_K_MAX_FRAC: float = 1.6   # max strike / S0; not specified
DUPIRE_N_K: int = 30         # strike grid points; not specified in paper

# SVI parameters for the synthetic surface (not specified in paper)
DUPIRE_SVI_A: float = 0.04
DUPIRE_SVI_B: float = 0.10
DUPIRE_SVI_RHO: float = -0.3
DUPIRE_SVI_M: float = 0.0
DUPIRE_SVI_XI: float = 0.15

# ---------------------------------------------------------------------------
# Exp 5 – Heston model
# ---------------------------------------------------------------------------

# Heston model parameters for synthetic calibration surface
# (Chapter 10, Section 10.4, page 153-154; paper does not specify values)
HESTON_TRUE_V0: float = 0.04     # initial variance (Y_0), Y_0 = sigma_0^2
HESTON_TRUE_KAPPA: float = 2.0   # mean-reversion speed
HESTON_TRUE_THETA: float = 0.04  # long-run variance
HESTON_TRUE_DELTA: float = 0.30  # vol-of-vol
HESTON_TRUE_RHO: float = -0.70   # correlation S-vol
HESTON_TRUE_R: float = 0.0       # risk-free rate; r=0 used in Chapter 10 for simplicity
HESTON_S0: float = 100.0
HESTON_T_GRID: list = [0.25, 0.5, 1.0, 1.5, 2.0]   # maturities for calibration grid
HESTON_K_FRAC: list = [0.8, 0.9, 1.0, 1.1, 1.2]    # strike / S0 for calibration grid
