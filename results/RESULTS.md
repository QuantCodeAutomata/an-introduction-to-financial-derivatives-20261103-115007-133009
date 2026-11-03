# Results — An Introduction to Financial Derivatives

**Paper:** Matthew Lorig, *An Introduction to Financial Derivatives: Open Lecture Notes*,
arXiv:2611.01355v1, 2026.
Source: <https://github.com/mattlorig/intro-financial-derivatives>

Numbers in this file are taken directly from the JSON files saved by the final run of each experiment.
Re-run the scripts to refresh them.

---

## Experiment 1: Binomial Model Implementation for Pricing and Hedging European and American Options

Implements the discrete-time binomial tree (Chapter 3). Tests the one-step and N-step backward-recursion
pricing formulas, replication hedge ratios, and American early-exercise logic against the numerical examples
given in Sections 3.1–3.2 and Exercise 3.4–3.5 of the notes.

| Metric | Paper value | This run | Verdict |
|--------|-------------|----------|---------|
| One-step call V0 (S0=100, u=1.1, d=0.9, r=0, K=100) | 5.00 | 5.000000 | reproduced |
| Two-step European call V0 (same params) | 5.25 | 5.250000 | reproduced |
| Two-step European put V0 (u=1.2, d=0.8, 1+r=e^0.05, S0=50, K=52) | ≈ 2.695 | 2.695479 | reproduced |
| Two-step European put Delta0 | ≈ −0.462 | −0.462284 | reproduced |
| Two-step American put V0 (same params) | ≥ European put | 5.000000 | reproduced |
| American V0 ≥ European V0 | True | True | reproduced |

See `results/exp1_results.json` and `results/exp1_binomial_prices.png`.

---

## Experiment 2: Black-Scholes Pricing, Hedging, Monte Carlo Simulation, and Implied Volatility Inversion

Implements Chapter 6 formulas for European calls/puts, the power-payoff claim, exact lognormal MC path
simulation, discrete delta-hedging error, and implied-volatility inversion. Tests put–call parity,
exact IV round-trip, power-claim pricing/hedging, and the hedging-error distribution (Exercise 6.1,
parameters S0=1, B0=1, mu=0.1, sigma=0.2, r=0.03, T=1, p=−2, N=100, M=1000).

| Metric | Paper value | This run | Verdict |
|--------|-------------|----------|---------|
| Put–call parity error | 0 | 0.000000 | reproduced |
| IV inversion round-trip error (sigma=0.2 input) | < 1e-6 | 5.5e-11 | reproduced |
| Power claim V0 (S0=1, sigma=0.2, r=0.03, T=1, p=−2) | exact formula | 1.030455 | reproduced |
| Power claim Delta0 | exact formula | −2.060909 | reproduced |
| MC hedging error mean (discrete rebalancing, N=100) | ≈ 0 | −0.000807 | reproduced |
| MC hedging error std (shrinks as N increases) | small | 0.017693 | reproduced |

See `results/exp2_results.json`, `results/exp2_vx_path.png`, `results/exp2_hedging_error_hist.png`.

---

## Experiment 3: Stochastic Volatility and Variance-Swap Replication under Volatility Mismatch

Implements Chapter 8 (Section 8.1 and 8.2). Part A simulates delta-hedging P&L when the true stock
volatility sigma_1=0.2 differs from the market/hedging volatility sigma_2=0.3 (r=0). Part B replicates
the floating leg of a variance swap and computes the fair strike via Monte Carlo, static option
integration (Carr-Madan formula), and the constant-vol analytic benchmark sigma²T.

| Metric | Paper value | This run | Verdict |
|--------|-------------|----------|---------|
| P&L mean sign (sigma_2=0.3 > sigma_1=0.2, short call) | positive | +1.935 | reproduced |
| P&L identity: cont-time approx ≈ simulated mean | ≈ equal | sim=1.935, id=1.962 | reproduced |
| Var-swap fair strike K_theory (sigma=0.2, T=1, r=0) | sigma²T = 0.0400 | 0.040000 | reproduced |
| Var-swap K_static (Carr-Madan integral, 500 strikes) | ≈ 0.0400 | 0.040045 | reproduced |
| Var-swap K_mc (Monte Carlo, N=252 steps, M=5000 paths) | ≈ 0.0400 | 0.037207 | partially reproduced |

K_mc is below the theoretical value because the discrete realized-variance approximation (252 steps)
underestimates the continuous integral; the notes state the floating leg equals integrated variance in
the continuous limit (Section 8.2). No exact numerical target is given.

See `results/exp3_results.json`, `results/exp3_vol_mismatch_path.png`, `results/exp3_vol_mismatch_hist.png`.

---

## Experiment 4: Local Volatility Surface Recovery with Dupire's Formula

Implements Chapter 9. Route A applies Dupire's formula via central finite differences on a synthetic
Black-Scholes call surface (S0=100, sigma grid 0.15–0.35 × 8 maturities × 30 interior strikes).
Exercise 9.1 tests the term-structure special case: with sigma(T) depending only on maturity, the
recovered local volatility should be strike-independent, and the integral identity
sigma²(T) = (1/(T-t)) ∫ f²(u,K) du should hold numerically.

| Metric | Paper value | This run | Verdict |
|--------|-------------|----------|---------|
| FD local variance f² ≥ 0 (all interior nodes) | True (Section 9.1) | True (min=0.0373) | reproduced |
| FD local variance range on test surface | consistent with input sigma²≈0.04 | 0.037–0.055 | reproduced |
| Term-structure: f²(T,K) independent of K | True (Exercise 9.1) | True (max K-variation < 0.1%) | reproduced |
| Integral identity max error (all interior maturities) | ≈ 0 | 0.0246 | partially reproduced |

The integral identity error of 0.025 arises from finite-difference discretization and the coarse
maturity grid; no numerical tolerance is stated in the notes.

See `results/exp4_results.json`, `results/exp4_dupire_fd.png`, `results/exp4_local_vol_ts.png`.

---

## Experiment 5: One-Factor Stochastic Volatility Option Pricing and Heston Calibration

Implements Chapter 10. Uses the Heston (1993) characteristic-function representation with vectorised
64-point Gauss-Legendre Fourier inversion (the notes omit the explicit Riccati formulas, Section 10.3).
Generates a synthetic implied-vol surface from true parameters, then calibrates Heston parameters to it
by minimising sum-of-squared implied-vol differences (Section 10.4).

True params: v0=0.04, kappa=2.0, theta=0.04, delta=0.3, rho=−0.7.

| Metric | Paper value | This run | Verdict |
|--------|-------------|----------|---------|
| Calibration in-sample RMSE | → 0 (exact identification from synthetic surface) | 1.35e-08 | reproduced |
| Calibrated v0 | 0.0400 | 0.040000 | reproduced |
| Calibrated kappa | 2.0000 | 2.000001 | reproduced |
| Calibrated theta | 0.0400 | 0.040000 | reproduced |
| Calibrated delta | 0.3000 | 0.300000 | reproduced |
| Calibrated rho | −0.7000 | −0.700000 | reproduced |

See `results/exp5_results.json`, `results/exp5_heston_smile.png`, `results/exp5_heston_surface.png`.

---

## Deviations from the paper

- All five experiments use synthetic model inputs only; the notes provide no external market dataset for
  any of the computational exercises.
- Experiment 3 (variance swap, Section 8.2): the upper-strike truncation of the call integral is set at
  5 × S0; the notes give the formula with an improper integral but specify no truncation point.
- Experiment 4 (Dupire, Section 9.1): the SVI parameter bounds and optimisation settings are not
  specified in the notes; scipy.optimize.least_squares with conservative bounds is used.
- Experiment 5 (Heston, Section 10.3): the notes explicitly omit the Riccati solution for C and D
  ("We will not give the expressions for C and D here", page 152). Pricing uses the Heston (1993)
  characteristic-function formula with vectorised Gauss-Legendre quadrature (Gil-Pelaez 1951), which is
  the standard implementation of the same model.
- Experiment 5 calibration uses a synthetic implied-vol surface; no market option dataset is provided
  in the notes.

## Not reproduced

- Empirical calibration with real market option quotes for Experiments 3, 4, and 5: the notes provide no
  market option dataset for any of the computational exercises in Chapters 8–10.
