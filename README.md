# An Introduction to Financial Derivatives

**Paper:** "An Introduction to Financial Derivatives" (lecture notes)

This repository reproduces the computational experiments from the lecture notes on financial derivatives, covering binomial trees, the Black-Scholes model, stochastic volatility, Dupire local volatility, and the Heston model.

---

## Experiments

| ID | Title |
|----|-------|
| exp_1 | Binomial model pricing and hedging of European and American options |
| exp_2 | Black-Scholes pricing, hedging, Monte Carlo simulation, and implied volatility inversion |
| exp_3 | Stochastic volatility and variance-swap replication under volatility mismatch |
| exp_4 | Local volatility surface recovery with Dupire's formula |
| exp_5 | One-factor stochastic volatility option pricing and Heston calibration |

---

## Repository layout

```
src/          Library modules (binomial, black_scholes, monte_carlo, implied_vol,
              variance_swap, dupire, heston, config)
exp/          Experiment scripts (exp1.py – exp5.py)
tests/        pytest suite
results/      Generated tables, plots, and RESULTS.md
data/         Data preparation code (only __init__.py; no external data used)
```

---

## Setup

```bash
pip install -r requirements.txt
```

No API keys are required; all experiments use model parameters only.

---

## Running experiments

All commands are run from the repository root.

```bash
# Binomial model (Chapter 3)
python -m exp.exp1

# Black-Scholes and Monte Carlo (Chapter 6)
python -m exp.exp2

# Variance swaps and vol-mismatch hedging (Chapter 8)
python -m exp.exp3

# Dupire local volatility (Chapter 9)
python -m exp.exp4

# Heston calibration (Chapter 10)
python -m exp.exp5
```

Run all five at once:

```bash
bash reproduce.sh
```

---

## Running tests

```bash
# Fast unit tests only (no market data required)
python -m pytest -m "not market_data" -q

# Full suite
python -m pytest -q
```

---

## Results

After running the experiments, results are saved in `results/`:

| File | Content |
|------|---------|
| `exp1_results.json` | Binomial prices and hedge ratios |
| `exp1_binomial_prices.png` | Comparison bar chart |
| `exp2_results.json` | BS prices, power-claim metrics, MC hedging error stats |
| `exp2_vx_path.png` | Sample path of V and X |
| `exp2_hedging_error_hist.png` | Histogram of terminal hedging errors |
| `exp3_results.json` | Vol-mismatch P&L stats, variance-swap fair strikes |
| `exp3_vol_mismatch_path.png` | Single path comparison of X and C |
| `exp3_vol_mismatch_hist.png` | Histogram of terminal P&L |
| `exp4_results.json` | Dupire local-vol surface statistics |
| `exp4_dupire_fd.png` | Finite-difference local-vol surface |
| `exp4_local_vol_ts.png` | Term-structure special case |
| `exp5_results.json` | Heston calibration parameters and RMSE |
| `exp5_heston_smile.png` | Market vs fitted smile |
| `exp5_heston_surface.png` | Calibrated implied-vol surface |
| `RESULTS.md` | Paper comparison with verdicts |

See `results/RESULTS.md` for a detailed comparison against the lecture notes.
