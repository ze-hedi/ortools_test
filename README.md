# OR-Tools Incremental Solver Benchmark

This project benchmarks **incremental solving** with OR-Tools' MathOpt API. The goal is to measure how different solvers handle a second solve after small perturbations to the model, compared to the cost of the initial solve.

## What we are testing

The benchmark loads an MPS model, solves it once (cold start), applies 1000 random perturbations, then solves again incrementally (warm start). This measures the solver's ability to reuse internal state (basis, factorization) across solves.

Three solvers are compared: **GLOP**, **HiGHS**, and **XPRESS**.

For each run, we collect:

| Metric | Description |
|---|---|
| **Init Build** | Time to construct the incremental solver (model handoff to the backend) |
| **1st Solve** | Wall-clock time for the initial (cold) solve |
| **2nd Solve** | Wall-clock time for the incremental (warm) solve after perturbation |
| **Update** | Time the solver spends processing model updates before the 2nd solve |
| **Simplex/Barrier iters** | Iteration counts for both solves |

## Perturbation modes

The third argument controls what kind of perturbation is applied between the two solves:

| Mode | What changes | What it tests |
|---|---|---|
| `coeff` | 1000 random constraint matrix coefficients are scaled by [0.9, 1.1] | Warm start after structural changes to the constraint matrix |
| `rhs` | 1000 random constraint bounds (upper and lower) are scaled by [0.9, 1.1] | Warm start after right-hand-side changes (basis likely still valid) |
| `obj` | 1000 random objective coefficients are scaled by [0.9, 1.1] | Warm start after objective changes (primal basis still feasible) |
| `basis` | No model change; the basis from the 1st solve is explicitly passed to the 2nd solve | Pure basis warm start (best-case scenario for incremental solving) |

## Installation

### 1. Build and install OR-Tools

Clone and build OR-Tools from source with CMake:

```bash
git clone https://github.com/google/or-tools.git
cd or-tools
cmake -S . -B build -DBUILD_DEPS=ON -DBUILD_EXAMPLES=OFF -DBUILD_SAMPLES=OFF
cmake --build build -j$(nproc)
cmake --install build --prefix /path/to/your/ortools_deps
```

### 2. Build this project

```bash
cd ortools_test
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/your/ortools_deps
cmake --build build -j$(nproc)
```

## Usage

```
./build/incremental_bench <mps_file> [num_runs=10] [coeff|rhs|obj|basis]
```

**Arguments:**

- `mps_file` -- path to an MPS model file
- `num_runs` -- number of repetitions for averaging (default: 10)
- `mode` -- perturbation mode: `coeff`, `rhs`, `obj`, or `basis` (default: `coeff`)

**Examples:**

```bash
# Run with default settings (coeff mode, 10 runs)
./build/incremental_bench model.mps

# Run basis warm start with 5 runs
./build/incremental_bench model.mps 5 basis
```

### Run all 4 modes

A convenience script runs all perturbation modes sequentially:

```bash
./run_incremental_bench.sh
```

Edit the `MPS` variable in the script to point to your model file.

## Output

Results are printed to stdout and written to a JSON file named `incremental_bench_<mode>.json`.
