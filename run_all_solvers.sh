#!/usr/bin/env bash
set -euo pipefail

EXE="$(dirname "$0")/build/bench_coeff_changes"

SOLVERS=(gscip xpress)

for solver in "${SOLVERS[@]}"; do
  echo "Running solver: $solver"
  "$EXE" "$solver"
done
