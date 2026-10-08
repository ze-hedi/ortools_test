#!/usr/bin/env bash
set -euo pipefail

MPS="/home/bouchehdahed/studies/clean_big_study/15-15_1000/sub/sub_15.mps"
BIN="./build/incremental_bench"

for mode in coeff rhs obj basis; do
  echo "=========================================="
  echo "  Running mode: $mode"
  echo "=========================================="
  "$BIN" "$MPS" 10 "$mode"
  echo ""
done
