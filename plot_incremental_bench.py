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

bars = ax.bar(x + offsets[1] * width, first_solve, width,
              label="1st Solve", color="#4CAF50")
for bar, val in zip(bars, first_solve):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8)

# 2nd Solve: update (bottom) + pure solve (top), stacked
pos = x + offsets[2] * width
ax.bar(pos, update_dur, width, label="Update (in 2nd Solve)", color="#F44336")
ax.bar(pos, solve_only, width, bottom=update_dur,
       label="Solve (in 2nd Solve)", color="#FF9800")
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

# --- Simplex iterations histogram ---
iters_png = f"incremental_bench_{suffix}_iters.png"

fig2, ax2 = plt.subplots(figsize=(10, 6))
width2 = 0.3
x2 = np.arange(len(solvers))

bars1 = ax2.bar(x2 - width2 / 2, first_simplex, width2,
                label="1st Solve", color="#4CAF50")
bars2 = ax2.bar(x2 + width2 / 2, second_simplex, width2,
                label="2nd Solve", color="#FF9800")

for bar, val in zip(bars1, first_simplex):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars2, second_simplex):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom", fontsize=9)

ax2.set_xlabel("Solver")
ax2.set_ylabel("Simplex Iterations")
ax2.set_title(f"Simplex Iterations — {mode_label} perturbation")
ax2.set_xticks(x2)
ax2.set_xticklabels(solvers)
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(iters_png, dpi=150)
plt.show()
print(f"Saved to {iters_png}")
