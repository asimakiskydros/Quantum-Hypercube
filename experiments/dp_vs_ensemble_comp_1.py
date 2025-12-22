"""
    Experiment #3: compares `cst_dfs` (Alg. 4) against an ensemble of `cst_max_smallest` (Alg. 1) and `cst_nearest` (Alg. 2)
    that selects the better result each time. Calculates mean soft and strict winrates of Alg. 4 and plots median-variance
    of relative cost difference between Alg. 4 and the ensemble.
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
            cost_dfs = sum(f(val) for val in cst_dfs(step=jump, width=n, f=f))
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


def plot_median_relative_improvement (axis, report, label):
    widths = list(range(1, len(report['per_width']) + 1))
    medians = [w['rel_stats']['median'] for w in report['per_width']]

    axis.plot(widths, medians, label=f'Improvement: {label}')
    axis.set_title(label)
    axis.legend([
        f"Alg. 4 mean winrate={report['mean_wr']:.2f}%\n"
        f"Alg. 4 mean strict winrate={report['mean_swr']:.2f}%"
    ], handlelength=0, handletextpad=0, fancybox=True)
    axis.fill_between(
        widths, 
        [w['rel_stats']['perc25'] for w in report['per_width']], 
        [w['rel_stats']['perc75'] for w in report['per_width']], 
        alpha=0.2, hatch='///', color='lightskyblue', edgecolor='cornflowerblue')
    

if __name__ == '__main__':
    max_width = 50
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True)
    for axis, (f, label) in zip(axes, [
        (lambda x: x * (x + 1) // 2, '$\mathcal{M}(m)=\mathcal{O}(m)$'),
        (lambda x: x * (x + 1) * (2 * x + 1) // 6, '$\mathcal{M}(m)=\mathcal{O}(m^2)$'),
        (lambda x: sum(log2(i) ** 3 for i in range(1, x + 1)), '$\mathcal{M}(m)=\mathcal{O}(log^3m)$') 
        ]):
        plot_median_relative_improvement(axis, compare(f, max_width), label)

    fig.suptitle('Relative cost difference between Alg. 4 and an ensemble of Alg. 1 and 2')
    fig.supxlabel('Host register width n')
    plt.tight_layout()
    plt.show()
