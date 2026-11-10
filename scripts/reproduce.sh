#!/usr/bin/env bash
# Run from the repository root in an activated Python environment.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
RESULTS_DIR="${1:-runs/desktop}"
python -m pytest -q
for seed in 0 1 2; do
  python -m relic.train --config configs/desktop.json --seed "$seed" --output "${RESULTS_DIR}/seed${seed}"
done
python -m relic.evaluate --results "$RESULTS_DIR"
python -m relic.verify_results --results "$RESULTS_DIR/evaluation"
python -m relic.report --results "$RESULTS_DIR"
