"""
    Experiment #1: performs flat comparison between `cst_max_smallest` (Alg. 1) and `cst_nearest` (Alg. 2)
    over raw absolute cost differences.
"""

# with this, the current experiment file is executable while being able to use src as a package
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.base_cst import cst_max_smallest
from src.greedy_cst import cst_nearest
from typing import Callable
from math import log2
import matplotlib.pyplot as plt


def compare (f: Callable, width: int):
    differences = []
    jumps = []

    for jump in range(2 ** width):
        cost_max_smallest = sum(f(val) for val in cst_max_smallest(step=jump, width=width))
        cost_nearest = sum(f(val) for val in cst_nearest(step=jump, width=width))
        # since better performace == smaller cost, `nearest` outperforms when the difference is negative
        differences.append(cost_nearest - cost_max_smallest)  
        jumps.append(jump)

    return {
        'jumps': jumps,
        'diffs': differences }


def plot_cost_difference (axis, report, label):
    colors = ['tab:blue' if diff >= 0 else 'tab:orange' for diff in report['diffs']]

    axis.bar(report['jumps'], report['diffs'], color=colors)
    axis.axhline(0, color='black', linewidth=1)
    axis.set_title(label)

    for jump, diff in zip(report['jumps'], report['diffs'], strict=True):
        if diff > 0:
            # annotate cases where `max_smallest` outperforms, by jump and absolute difference
            axis.text(jump, diff, f's={jump}, d=+{diff:.2f}', ha='center', va='bottom', fontsize=7)


if __name__ == '__main__':
    width = 6
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True)
    funcs = [
        (lambda x: x * (x + 1) // 2, '$\mathcal{M}(m)=\mathcal{O}(m)$'),
        (lambda x: x * (x + 1) * (2 * x + 1) // 6, '$\mathcal{M}(m)=\mathcal{O}(m^2)$'),
        (lambda x: sum(log2(i) ** 3 for i in range(1, x + 1)), '$\mathcal{M}(m)=\mathcal{O}(log^3m)$')]

    for axis, (f, label) in zip(axes, funcs):
        plot_cost_difference(axis, compare(f, width), label)
    
    fig.suptitle('Cost difference between Alg. 1 and 2')
    fig.supxlabel('Shift step s')
    plt.tight_layout()
    plt.show()