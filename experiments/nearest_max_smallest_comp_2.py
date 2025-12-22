"""
    Experiment #2: performs better comparison between `cst_max_smallest` (Alg. 1) and `cst_nearest` (Alg. 2)
    over Alg. 1 winrate and relative cost differences (percentage-based over the cost of the other).
"""

# with this, the current experiment file is executable while being able to use src as a package
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.base_cst import cst_max_smallest
from src.greedy_cst import cst_nearest
from src.mytools import sample, stats
from math import log2
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np


def compare (f: Callable, max_width: int, max_samples=10_000, seed=42):
    per_width = []
    
    # compute and compare costs between both methods for all shifts
    for n in range(1, max_width + 1):
        hits, rel_diffs_maxsmall, rel_diffs_nearest = 0, [], []
        samples = sample(n, max_samples, seed)
        for jump in samples:
            cost_max_smallest = sum(f(val) for val in cst_max_smallest(step=jump, width=n))
            cost_nearest = sum(f(val) for val in cst_nearest(step=jump, width=n))
            if cost_max_smallest < cost_nearest:
                hits += 1  # hits are instances where max smallest wins
                rel_diffs_maxsmall.append((cost_nearest - cost_max_smallest) / cost_nearest if cost_nearest != 0 else np.nan)
            else:
                rel_diffs_nearest.append((cost_max_smallest - cost_nearest) / cost_max_smallest if cost_max_smallest != 0 else np.nan)
    
        per_width.append({
            'wins': hits * 100 / len(samples),  # calculate the hit percentage for this width
            'rel_stats_maxsmall': stats(rel_diffs_maxsmall),  # save statistical data for the recorded differences
            'rel_stats_nearest':  stats(rel_diffs_nearest) })
    return per_width


def plot_winrate_per_width (axis, report, label):
    wins = [w['wins'] for w in report]
    maxwins, minwins, startwins = max(wins), min(i for i in wins if i > 0), next(i for i in wins if i > 0)

    for i, bar in enumerate(axis.bar(list(range(1, len(wins) + 1)), wins)):
        if wins[i] in (maxwins, minwins, startwins): 
            axis.text(bar.get_x() + bar.get_width() * 0.5, bar.get_height() + 0.02, f'{wins[i]:.1f}%', ha='center', va='bottom', fontsize=7)

    axis.set_title(label)


def plot_median_relative_improvement (axis, report, label):
    widths = list(range(1, len(report) + 1))
    
    med_max_smallest    = [w['rel_stats_maxsmall']['median'] for w in report]
    perc25_max_smallest = [w['rel_stats_maxsmall']['perc25'] for w in report]
    perc75_max_smallest = [w['rel_stats_maxsmall']['perc75'] for w in report]

    med_nearest    = [w['rel_stats_nearest']['median'] for w in report]
    perc25_nearest = [w['rel_stats_nearest']['perc25'] for w in report]
    perc75_nearest = [w['rel_stats_nearest']['perc75'] for w in report]

    axis.plot(
        widths, med_max_smallest, 
        linestyle='-', marker='o', markersize=5, color='tab:orange', label='Alg. 1 win')
    axis.fill_between(widths, perc25_max_smallest, perc75_max_smallest, alpha=0.2, hatch='///', color='navajowhite', edgecolor='sandybrown')
    axis.plot(
        widths, med_nearest, 
        linestyle='--', marker='s', markersize=5, color='tab:blue', label='Alg. 2 win')
    axis.fill_between(widths, perc25_nearest, perc75_nearest, alpha=0.2, hatch='\\\\\\', color='lightskyblue', edgecolor='cornflowerblue')
    axis.set_title(label)
    axis.legend(loc='upper left')


if __name__ == '__main__':
    max_width = 50
    funcs = [
        (lambda x: x * (x + 1) // 2, '$\mathcal{M}(m)=\mathcal{O}(m)$'),
        (lambda x: x * (x + 1) * (2 * x + 1) // 6, '$\mathcal{M}(m)=\mathcal{O}(m^2)$'),
        (lambda x: sum(log2(i) ** 3 for i in range(1, x + 1)), '$\mathcal{M}(m)=\mathcal{O}(log^3m)$') ]
    figs = [
        plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True),
        plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True) ]
    plotters = [
        plot_winrate_per_width, 
        plot_median_relative_improvement]
    
    for i, (f, label) in enumerate(funcs):
        report = compare(f, max_width)
        for plotter, (fig, axes) in zip(plotters, figs, strict=True):
            plotter(axes[i], report, label)

    figs[0][0].suptitle('Alg. 1 winrate% per width')
    figs[1][0].suptitle('Median relative improvement per width on wins')
    figs[0][0].supxlabel('Host register width n')
    figs[0][0].supxlabel('Host register width n')
    plt.tight_layout()
    plt.show()