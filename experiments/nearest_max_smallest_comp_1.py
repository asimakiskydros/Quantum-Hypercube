"""
Experiment #1: performs flat comparison between `MaxSmallest` (Alg. 1) and `Nearest` (Alg. 2)
over raw absolute cost differences.
Relevant to:  https://doi.org/10.1002/qute.70445
"""

# with this, the current experiment file is executable while being able to use src as a package
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cycle_shifts import MaxSmallest, Nearest
from src.mytools import cost_models, latexify
import matplotlib.pyplot as plt


if __name__ == '__main__':
    width = 6
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True)
    nearest, maxsmallest = Nearest(), MaxSmallest()

    for ax, (label, f) in zip(axes, cost_models(width).items()):
        # since better performace == smaller cost, `nearest` outperforms when the difference is negative
        differences = [nearest.cost(jump, width, f) - maxsmallest.cost(jump, width, f) for jump in range(1 << width)]
        # plot cost differences
        ax.bar(range(1 << width), differences, color=['tab:blue' if diff >= 0 else 'tab:orange' for diff in differences])
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(latexify(label))
        # annotate cases where `max_smallest` outperforms, by jump and absolute difference
        for jump, diff in enumerate(differences):
            if diff > 0: ax.text(jump, diff, f's={jump}, d=+{diff:.2f}', ha='center', va='bottom', fontsize=7)
                
    fig.suptitle('Cost difference between Alg. 1 and 2')
    fig.supxlabel('Shift step s')
    plt.tight_layout()
    plt.show()
