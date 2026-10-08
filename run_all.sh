#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"

"$SCRIPT_DIR/run_all_solvers.sh"
"$SCRIPT_DIR/run_all_solvers_mpsolver.sh"
