"""
Experiment #2: performs better comparison between `MaxSmallest` (Alg. 1) and `Nearest` (Alg. 2)
over Alg. 1 winrate and relative cost differences (percentage-based over the cost of the other).
Relevant to:  https://doi.org/10.1002/qute.70445
"""

# with this, the current experiment file is executable while being able to use src as a package
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cycle_shifts import MaxSmallest, Nearest
from src.mytools import sample, Stats, Report, cost_models, latexify
from typing import Callable, Sequence
from numpy import nan, arange
import matplotlib.pyplot as plt


def compare (l: str, f: Callable, max_width: int, max_samples=10_000, seed=42, verbose=True):
    """
    Calculates the growth of the relative improvement of `MaxSmallest` against `Nearest` with the width
    and vice-versa.
    """
    report = Report(label=l)
    hit = lambda a, b: a < b  # hit when first has less cost
    ri = lambda a, b: (a - b) / a if a else nan  # relative improvement

    for n in range(1, max_width+1):
        samples = sample(n, max_samples, seed)
        # collect costs per method for every sampled jump
        costs = list(zip(MaxSmallest().cost(samples, n, f), Nearest().cost(samples, n, f)))
        # identify cases where maxsmallest is better (less cost) and rest
        hits = [ri(cn, cm) for cm, cn in costs if hit(cm, cn)]
        misses = [ri(cm, cn) for cm, cn in costs if not hit(cm, cn)]
        report.append(winrate=len(hits) * 100 / len(samples), improvement=Stats(hits), other=Stats(misses))

    if verbose: print(f'Done with {l}')

    return report
    

def plot_winrate_per_width (reports: Sequence[Report]):
    """
    Generates a multi-bar graph of the win rate of Alg. 1 per width per tested cost function.
    """
    fig, axes = plt.subplots(nrows=1, ncols=len(reports), figsize=(5 * len(reports), 5), sharex=True)
    
    for ax, report in zip(axes, reports, strict=True):
        ax.set_title(latexify(report.label))
        bars = ax.bar(range(1, len(report.wins)+1), report.wins)
        pairs = [(wr, bar) for wr, bar in zip(report.wins, bars, strict=True) if wr]

        for wr, bar in (pairs[0], max(pairs, key=lambda x: x[0]), min(pairs, key=lambda x: x[0])):
            # annotate first non zero, max and min non zero bars
            ax.text(
                bar.get_x() + bar.get_width() * 0.5, 
                bar.get_height() + 0.02, 
                f'{wr:.1f}%', ha='center', va='bottom', fontsize=7)
    
    fig.suptitle('Alg. 1 winrate% per width')
    fig.supxlabel('Host register width n')
    return fig


def plot_median_relative_improvement (reports: Sequence[Report]):
    """
    Generates a multi-line graph of the median relative improvement of each method per width, surrounded by its 25th-75th 
    percentile range, per tested cost function.
    """
    fig, axes = plt.subplots(nrows=1, ncols=len(reports), figsize=(5 * len(reports), 5), sharex=True)
    for ax, report in zip(axes, reports, strict=True):
        w = arange(1, len(report.wins) + 1)
        label = latexify(report.label)
        format = (
            { 'linestyle': '-',  'marker': 'o', 'color': 'tab:orange', 'fpr': ('///', 'navajowhite', 'sandybrown') },
            { 'linestyle': '--', 'marker': 's', 'color': 'tab:blue',   'fpr': ('\\\\\\', 'lightskyblue', 'cornflowerblue') },
        )
        for i, frmt in enumerate(format):
            fpr = frmt.pop('fpr', ())
            report.plot_median_curve(ax, w, use_other=i, **frmt, label=label)
            report.plot_25_75_percentile_range(ax, w, *fpr, use_other=i, label=label)
            
        ax.set_title(label)
        ax.legend(loc='upper left')

    fig.suptitle('Median relative improvement per width on wins')
    fig.supxlabel('Host register width n')
    return fig


if __name__ == '__main__':
    max_width = 50
    reports = [compare(*f, max_width) for f in cost_models(max_width).items()]
    f1 = plot_winrate_per_width(reports)
    f2 = plot_median_relative_improvement(reports)
    plt.tight_layout()
    plt.show()
