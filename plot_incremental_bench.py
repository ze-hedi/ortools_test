#!/usr/bin/env python3
import json
import sys
import matplotlib.pyplot as plt
import numpy as np

json_file = sys.argv[1] if len(sys.argv) > 1 else "incremental_bench_coeff.json"
# Extract suffix (_coeff or _rhs) from filename
base = json_file.rsplit(".", 1)[0]  # e.g. "incremental_bench_coeff"
suffix = base.rsplit("_", 1)[-1]    # e.g. "coeff" or "rhs"
png_file = f"incremental_bench_{suffix}.png"

with open(json_file) as f:
    data = json.load(f)

solvers = list(data.keys())

# 3 bars per solver: Initial Build, 1st Solve, 2nd Solve (with Update stacked inside)
x = np.arange(len(solvers))
width = 0.22
offsets = np.array([-1, 0, 1])

initial_build = [data[s]["initial_build_ms"] for s in solvers]
first_solve = [data[s]["first_solve_ms"] for s in solvers]
second_solve = [data[s]["second_solve_ms"] for s in solvers]
update_dur = [data[s]["update_duration_ms"] for s in solvers]
solve_only = [s2 - u for s2, u in zip(second_solve, update_dur)]
first_simplex = [int(data[s].get("first_simplex_iters", 0)) for s in solvers]
first_barrier = [int(data[s].get("first_barrier_iters", 0)) for s in solvers]
second_simplex = [int(data[s].get("second_simplex_iters", 0)) for s in solvers]
second_barrier = [int(data[s].get("second_barrier_iters", 0)) for s in solvers]

fig, ax = plt.subplots(figsize=(10, 6))

# Initial Build
bars = ax.bar(x + offsets[0] * width, initial_build, width,
              label="Initial Build", color="#2196F3")
for bar, val in zip(bars, initial_build):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8)

# 1st Solve — build legend label with iteration counts per solver
first_iters_parts = []
for s, si, bi in zip(solvers, first_simplex, first_barrier):
    parts = []
    if si: parts.append(f"S:{si}")
    if bi: parts.append(f"B:{bi}")
    first_iters_parts.append(f"{s}: {', '.join(parts)}" if parts else f"{s}: 0")
first_solve_label = "1st Solve\n  " + "\n  ".join(first_iters_parts)
bars = ax.bar(x + offsets[1] * width, first_solve, width,
              label=first_solve_label, color="#4CAF50")
for bar, val in zip(bars, first_solve):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8)

# 2nd Solve: update (bottom) + pure solve (top), stacked
pos = x + offsets[2] * width
ax.bar(pos, update_dur, width, label="Update (in 2nd Solve)", color="#F44336")

second_iters_parts = []
for s, si, bi in zip(solvers, second_simplex, second_barrier):
    parts = []
    if si: parts.append(f"S:{si}")
    if bi: parts.append(f"B:{bi}")
    second_iters_parts.append(f"{s}: {', '.join(parts)}" if parts else f"{s}: 0")
solve_only_label = "Solve (in 2nd Solve)\n  " + "\n  ".join(second_iters_parts)
ax.bar(pos, solve_only, width, bottom=update_dur,
       label=solve_only_label, color="#FF9800")
for xi, total, upd in zip(pos, second_solve, update_dur):
    ax.text(xi, total + 0.5, f"{total:.1f}", ha="center", va="bottom", fontsize=8)
    if upd > 0.5:
        ax.text(xi, upd / 2, f"{upd:.1f}", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold")

ax.set_xlabel("Solver")
ax.set_ylabel("Time (ms)")
mode_labels = {"coeff": "Coefficient", "rhs": "RHS", "obj": "Objective", "basis": "Basis warm-start"}
mode_label = mode_labels.get(suffix, suffix)
ax.set_title(f"Incremental Solver Benchmark — {mode_label} perturbation")
ax.set_xticks(x)
ax.set_xticklabels(solvers)
ax.legend(fontsize=8, loc="upper right", prop={"family": "monospace", "size": 8})
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(png_file, dpi=150)
plt.show()
print(f"Saved to {png_file}")
