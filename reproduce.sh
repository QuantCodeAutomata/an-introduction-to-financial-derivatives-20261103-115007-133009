#!/usr/bin/env bash
# Run all experiments and the test suite from the repository root.
set -e
python -m pytest -m "not market_data" -q
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
