#!/usr/bin/env bash
set -euo pipefail

for mode in coeff rhs obj basis; do
  json="incremental_bench_${mode}.json"
  if [ ! -f "$json" ]; then
    echo "Skipping $mode: $json not found"
    continue
  fi
  echo "Plotting $mode..."
  python3 plot_incremental_bench.py "$json"
done
