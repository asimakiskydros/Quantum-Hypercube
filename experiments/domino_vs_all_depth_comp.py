# with this, the current experiment file is executable while being able to use src as a package
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cycle_shifts import CST, Domino
from src.mytools import sample, Report
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from numpy import arange
import matplotlib.pyplot as plt


def compare (max_width: int, max_samples=10_000, seed=42, verbose=True):
    """
    Calculates the winrate and relative improvement of Domino on the depth over the rest.
    """
    reports = defaultdict(lambda: Report())

    with Pool(cpu_count()) as pool:
        for n in range(1, max_width + 1):
            samples = sample(n, max_samples, seed)
            results = {
                method.name: pool.starmap(method.depth, [(jump, n) for jump in samples])
                for method in CST.methods()
            }
            result_dom = results.pop(Domino.__name__)
            for method, result in results.items():
                reports[method].process(result, result_dom, num_samples=len(samples))

            if verbose: print(f'Done with width {n}')

    return reports    


if __name__ == '__main__':
    max_width = 50
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), sharex=True)

    for ax, (method, report) in zip(axes, compare(max_width).items()):
        w = arange(1, max_width + 1)
        report.plot_median_curve(ax, w)
        report.plot_25_75_percentile_range(ax, w)
        report.add_winrates_on_legend(ax)
        ax.set_title(f'Against {method}')

    fig.suptitle('Median relative improvement of Domino on circuit depth per algorithm per width')
    fig.supxlabel('Host register width n')
    plt.tight_layout()
    plt.show()
