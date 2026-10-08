import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [10, 100, 200, 500]

# Data per solver
solvers = {
    'GLOP': {
        'set_avg_us': [39.3, 238.2, 461.2, 1126.4],
        'solve_avg_ms': [6084.38, 6157.42, 6029.93, 6157.34],
        'baseline_avg_ms': 6126.4,
    },
    'HiGHS': {
        'set_avg_us': [48, 263, 489.9, 1129.7],
        'solve_avg_ms': [18404, 17803.1, 18775.3, 18421.5],
        'baseline_avg_ms': 16784,
    },
    'PDLP': {
        'set_avg_us': [44.3, 251, 476.7, 1167.9],
        'solve_avg_ms': [18341.2, 18119.8, 18276.4, 18535.8],
        'baseline_avg_ms': 18283.6,
    },
    'GSCIP': {
        'set_avg_us': [46.5, 282.5, 520, 2148],
        'solve_avg_ms': [337005, 346940, 349879, 397047],
        'baseline_avg_ms': 306059,
    },
}

colors = {'GLOP': 'tab:blue', 'HiGHS': 'tab:orange', 'PDLP': 'tab:green', 'GSCIP': 'tab:red'}

out_dir = '/home/bouchehdahed/code/ortools_test'

# --- Plot 1: set_coefficient time ---
plt.figure(figsize=(8, 5))
for name, data in solvers.items():
    plt.plot(batch_sizes, data['set_avg_us'], 'o-', color=colors[name],
             linewidth=2, markersize=8, label=name)
    for x, y in zip(batch_sizes, data['set_avg_us']):
        plt.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                     xytext=(0, 12), ha='center', fontsize=8, color=colors[name])

xs = np.linspace(batch_sizes[0], batch_sizes[-1], 100)
plt.plot(xs, xs, '--', color='tab:gray', linewidth=1, label='y = x')
plt.legend()
plt.xlabel('Batch size (number of coefficient changes)')
plt.ylabel('set_coefficient avg time (µs)')
plt.title('set_coefficient time vs batch size')
plt.xticks(batch_sizes)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{out_dir}/bench_plot_set.png', dpi=150)
print("Saved bench_plot_set.png")

# --- Plot 2: solve time ---
plt.figure(figsize=(8, 5))
for name, data in solvers.items():
    all_ms = [data['baseline_avg_ms']] + data['solve_avg_ms']
    all_x = [0] + batch_sizes
    plt.plot(all_x, all_ms, 'o-', color=colors[name], linewidth=2, markersize=8, label=name)
    for x, y in zip(all_x, all_ms):
        plt.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                     xytext=(0, 12), ha='center', fontsize=8, color=colors[name])

plt.xlabel('Batch size (number of coefficient changes)')
plt.ylabel('Solve avg time (ms)')
plt.title('Solve time vs batch size')
plt.xticks([0] + batch_sizes, ['baseline'] + [str(b) for b in batch_sizes])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{out_dir}/bench_plot_solve.png', dpi=150)
print("Saved bench_plot_solve.png")
