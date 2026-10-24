"""
Experiment #5: Fidelity comparison between methods.
Relevant to:  https://doi.org/10.1002/qute.70445
"""

# with this, the current experiment file is executable while being able to use src as a package
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cycle_shifts import CST
from src.mytools import CustomNoiseModel
from numpy import arange
import matplotlib.pyplot as plt


if __name__ == '__main__':
    """
    Calculates and plots the median evolution of the fidelity per width, per selected noisy backend.
    Compares the first three `CST` versions, i.e. `MaxSmallest`, `Nearest` and `DP_DFS`.
    """
    results = CustomNoiseModel.calculate_fidelities(
        CST.methods(k=3),
        max_width=10,
        date='21/4/26'
    )
    labels = ('1', '2', '4*')
    fig, axes = plt.subplots(nrows=len(results), ncols=len(labels), figsize=(7, 5))

    for ax_col, (backend, digest) in zip(axes, results.items()):
        for ax, report, title in zip(ax_col, digest.values(), labels):
            w = arange(1, len(report.improvements) + 1)
            report.plot_median_curve(ax, w)
            report.plot_25_75_percentile_range(ax, w)
            report.median_bounds_on_yaxis(ax)
            if ax is ax_col[0]: ax.set_ylabel(backend)       # label rows according to noise backend used
            if ax in axes[0]: ax.set_title(f'Alg. {title}')  # label columns according to algorithm used
    
    fig.suptitle('Median fidelity evolution per algorithm per width')
    fig.supxlabel('Host register width n')
    plt.tight_layout()
    plt.show()
