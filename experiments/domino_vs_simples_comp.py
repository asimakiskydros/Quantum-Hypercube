# with this, the current experiment file is executable while being able to use src as a package
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cycle_shifts import MaxSmallest, Nearest, Domino
from src.mytools import sample, Report
from numpy import arange
import matplotlib.pyplot as plt


def compare (max_width: int, max_samples=10_000, seed=42):
    """
    Calculates the winrate and relative improvement of `Domino` per width compared to `MaxSmallest` and `Nearest`.
    """
    # in this context, the labels mean "against maxsmall" or "against nearest"
    reports = { MaxSmallest(): Report(), Nearest(): Report() }
    domino = Domino()

    for n in range(1, max_width + 1):
        samples = sample(n, max_samples, seed)
        cost_dom = domino.cost(samples, n)

        for cst, report in reports.items(): 
            report.process(cst.cost(samples, n), cost_dom, num_samples=len(samples))

    return reports


if __name__ == '__main__':
    max_width = 50
    reports = compare(max_width)
    fig, axes = plt.subplots(nrows=1, ncols=len(reports), figsize=(5 * len(reports), 5), sharex=True)

    for ax, (method, report) in zip(axes, reports.items()):
        ax.set_title(f'Against {method.name}')
        report.plot_median_curve(ax, w := arange(1, max_width + 1))
        report.plot_25_75_percentile_range(ax, w)
        report.add_winrates_on_legend(ax)
    
    fig.suptitle('Median relative improvement of Domino per width')
    fig.supxlabel('Host register width n')
    plt.tight_layout()
    plt.show()
