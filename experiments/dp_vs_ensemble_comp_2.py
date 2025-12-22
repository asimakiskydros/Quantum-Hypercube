"""
    Experiment #4: performs the same comparison as in Expr. 3 but `decompose` (Alg. 3) now internally uses the 
    heuristic f(n-p) = n-p. Shift sequence costs are still computed using realistic cost models.
"""

# with this, the current experiment file is executable while being able to use src as a package
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.base_cst import cst_max_smallest
from src.greedy_cst import cst_nearest
from src.dp_cst import cst_dfs
from src.mytools import sample, stats
from math import log2
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np


def compare (f: Callable, max_width: int, max_samples=10_000, seed=42):
    per_width = []

    for n in range(1, max_width + 1):
        hits, strict_hits, rel_diffs = 0, 0, []
        samples = sample(n, max_samples, seed)
        for jump in samples:
            cost_dfs = sum(f(val) for val in cst_dfs(step=jump, width=n, f=lambda x: x))  # heuristic f
            cost_ensemble = min(
                sum(f(val) for val in cst_max_smallest(step=jump, width=n)), 
                sum(f(val) for val in cst_nearest(step=jump, width=n)))

            if cost_dfs < cost_ensemble:
                strict_hits += 1
            if cost_dfs <= cost_ensemble:
                hits += 1
            
            rel_diffs.append(abs(cost_ensemble - cost_dfs) / cost_ensemble if cost_ensemble != 0 else np.nan)

        per_width.append({
            'wins': hits * 100 / len(samples),
            'strict_wins': strict_hits * 100 / len(samples),
            'rel_stats': stats(rel_diffs) })
    return {
        'per_width': per_width,
        'mean_wr': np.nanmean([w['wins'] for w in per_width]),
        'mean_swr': np.nanmean([w['strict_wins'] for w in per_width]), }


def plot_median_relative_improvement (report, label, styles):
    widths = list(range(1, len(report['per_width']) + 1))
    medians = [w['rel_stats']['median'] for w in report['per_width']]

    plt.plot(widths, medians, label=label,
        linestyle=styles[0], marker=styles[1], markersize=5, color=styles[2])
    plt.fill_between(
        widths, 
        [w['rel_stats']['perc25'] for w in report['per_width']], 
        [w['rel_stats']['perc75'] for w in report['per_width']], 
        label=label, alpha=0.2, hatch=styles[3], color=styles[4], edgecolor=styles[5])


if __name__ == '__main__':
    max_width = 50
    funcs = [
        (lambda x: x * (x + 1) // 2, '$\mathcal{M}(m)=\mathcal{O}(m)$'),
        (lambda x: x * (x + 1) * (2 * x + 1) // 6, '$\mathcal{M}(m)=\mathcal{O}(m^2)$'), 
        (lambda x: sum(log2(i) ** 3 for i in range(1, x + 1)), '$\mathcal{M}(m)=\mathcal{O}(log^3m)$') ]
    styles = [
        ['-', 'o', 'tab:orange', '///', 'navajowhite', 'sandybrown'],
        ['--', 's', 'tab:blue', '\\\\\\', 'lightskyblue', 'cornflowerblue'],
        ['-.', '^', 'tab:green', 'ooo', 'lightgreen', 'seagreen'] ]


    plt.figure(figsize=(9, 5))

    for (f, label), style in zip(funcs, styles, strict=True):
        report = compare(f, max_width)
        plot_median_relative_improvement(report, label, style)
        print(f"{label}: Alg. 4 mean winrate={report['mean_wr']:.2f}%")
        print(f"{label}: Alg. 4 mean strict winrate={report['mean_swr']:.2f}%")

    plt.title(f'Relative cost difference between Alg. 4(heuristic) and an ensemble of Alg. 1 and 2')
    plt.xlabel('Host register width n')
    plt.legend()
    plt.tight_layout()
    plt.show()
