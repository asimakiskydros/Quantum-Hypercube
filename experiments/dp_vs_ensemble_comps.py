"""
Experiments #3 and #4: Comparing DP-DFS to an ensemble of MaxSmallest and Nearest.
Relevant to:  https://doi.org/10.1002/qute.70445
"""

# with this, the current experiment file is executable while being able to use src as a package
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cycle_shifts import DP_DFS, MaxSmallest, Nearest
from src.mytools import sample, Report, cost_models, latexify
from typing import Callable, Sequence
from numpy import mean, arange
import matplotlib.pyplot as plt


def compare (l: str, f: Callable, max_width: int, max_samples=10_000, seed=42, use_heuristic=False, verbose=True):
    """
    Computes the winrate and relative improvement per width of `DP_DFS` against an ensemble of `MaxSmallest` and `Nearest`.
    """
    report = Report(label=l)
    foo = f if not use_heuristic else None

    for n in range(1, max_width + 1):
        samples = sample(n, max_samples, seed)
        cost_dfs = DP_DFS().cost(samples, n, f, foo)
        cost_ensemble = [
            min(a, b) for a, b in zip(
                MaxSmallest().cost(samples, n, f), 
                Nearest().cost(samples, n, f), strict=True)
        ]
        report.process(cost_ensemble, cost_dfs, num_samples=len(samples))
    
    if verbose: print(f'Done with {l}')

    return report


def plot_median_relative_improvement (reports: Sequence[Report], format: Sequence=None, media=None):
    """
    Generates a (multi-)line graph of the median relative improvement of `DP_DFS` against the ensemble per width,
    surrounded by its 25th-75th percentile range.
    """
    format = [{}] * len(reports) if format is None else format
    media  = [plt] * len(reports) if media is None else media
    
    for medium, report, frmt in zip(media, reports, format, strict=True):
        fpr = frmt.pop('fpr', ())
        w = arange(1, len(report.wins) + 1)
        label = latexify(report.label)
        report.plot_median_curve(medium, w, **frmt, label=label)
        report.plot_25_75_percentile_range(medium, w, *fpr, label=label)
        if frmt:
            print(f"{report.label}:")
            print(f"- Alg. 4 mean winrate={mean(report.wins):.2f}%")
            print(f"- Alg. 4 mean strict winrate={mean(report.strict_wins):.2f}%")
        else:
            medium.set_title(label)
            report.add_winrates_on_legend(medium)
 

if __name__ == '__main__':
    max_width = 50
    styles = (
        { 'linestyle': '-',  'marker': 'o', 'color': 'tab:orange', 'fpr': ('///', 'navajowhite', 'sandybrown') },
        { 'linestyle': '--', 'marker': 's', 'color': 'tab:blue',   'fpr': ('\\\\\\', 'lightskyblue', 'cornflowerblue') },
        { 'linestyle': '-.', 'marker': '^', 'color': 'tab:green',  'fpr': ('ooo', 'lightgreen', 'seagreen') },
    )
    models = cost_models(max_width)
    """
    Experiment #3: compares `DP_DFS` (Alg. 4) against an ensemble of `MaxSmallest` (Alg. 1) and `Nearest` (Alg. 2)
    that selects the better result each time. Calculates mean soft and strict winrates of Alg. 4 and plots median-variance
    of relative cost difference between Alg. 4 and the ensemble.
    """
    reports = [compare(*f, max_width) for f in models.items()]
    f1, axes = plt.subplots(nrows=1, ncols=len(reports), figsize=(5 * len(reports), 5), sharex=True)
    plot_median_relative_improvement(reports, media=axes)
    f1.suptitle('Relative cost difference between Alg. 4 and an ensemble of Alg. 1 and 2')
    f1.supxlabel('Host register width n')

    """
    Experiment #4: performs the same comparison as in Expr. 3 but `decompose` (Alg. 3) now internally uses the 
    heuristic f(n-p) = n-p. Shift sequence costs are still computed using realistic cost models.
    """
    reports = [compare(*f, max_width, use_heuristic=True) for f in models.items()]
    f2 = plt.figure(figsize=(10, 5))
    plot_median_relative_improvement(reports, format=styles)
    plt.title('Relative cost difference between Alg. 4(heuristic) and an ensemble of Alg. 1 and 2')
    plt.xlabel('Host register width n')
    plt.legend()

    plt.tight_layout()
    plt.show()
