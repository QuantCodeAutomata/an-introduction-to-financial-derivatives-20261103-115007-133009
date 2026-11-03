# AGENTS.md — Guide for coding agents and contributors

## What this repository implements

Computational experiments from the lecture notes "An Introduction to Financial Derivatives", covering:

- **Chapter 3 (exp_1):** Binomial tree pricing and hedging of European and American options.
- **Chapter 6 (exp_2):** Black-Scholes closed-form pricing, power claims, Monte Carlo delta-hedging simulation, and implied volatility inversion.
- **Chapter 8 (exp_3):** Delta-hedging under volatility mismatch; variance-swap floating-leg replication and fair-strike computation from static option integration.
- **Chapter 9 (exp_4):** Dupire local-volatility surface recovery via finite-difference and SVI-smoothed differentiation of a call-price surface.
- **Chapter 10 (exp_5):** Heston stochastic-volatility model pricing and calibration to an implied-volatility surface.

## Repository layout

```
src/           Flat library modules (no subdirectories)
  binomial.py      CRR binomial tree, one-step and N-step pricing, American options, calibration
  black_scholes.py Black-Scholes formulas, Greeks, power-claim pricing
  config.py        All experiment parameters as module-level UPPER_CASE constants
  dupire.py        Dupire local-vol formula, FD and SVI routes
  heston.py        Heston model pricing and calibration
  implied_vol.py   Implied-volatility inversion via Brent's method
  monte_carlo.py   Exact lognormal path simulation, replication portfolio update
  variance_swap.py Variance-swap replication and Carr-Madan fair-strike integration

exp/           Experiment scripts
  exp1.py  exp2.py  exp3.py  exp4.py  exp5.py

tests/         pytest files (all named test_*.py)
  conftest.py  test_binomial.py  test_black_scholes.py
  test_dupire.py  test_heston.py  test_variance_swap.py

results/       Generated artifacts (PNGs, JSON, RESULTS.md) — committed
data/          Data preparation code; only __init__.py (no external data used)
```

## Setup

```bash
pip install -r requirements.txt
```

No API keys are required; all experiments use model parameters only.

## Commands

### Tests

```bash
# Fast (no market data)
python -m pytest -m "not market_data" -q

# Full suite
python -m pytest -q
```

### Experiments

```bash
python -m exp.exp1   # Binomial model
python -m exp.exp2   # Black-Scholes + Monte Carlo
python -m exp.exp3   # Variance swap / vol mismatch
python -m exp.exp4   # Dupire local vol
python -m exp.exp5   # Heston calibration
```

```bash
bash reproduce.sh    # Tests + all five experiments
```

## Conventions

- **All parameters** live in `src/config.py` as module-level `UPPER_CASE` variables with comments giving their meaning, unit, and paper source.
- **All data loading** is done through `data/data_loader.py` (no such file is needed here because no market data is used).
- **No secrets** in any file. If market data were added, API keys would go in a `.env` file listed in `.gitignore`, with placeholders in `.env.example`.
- After a rerun, experiment scripts overwrite the files in `results/`. The numbers and verdicts in `results/RESULTS.md` must then be checked against the new files and updated by hand.
