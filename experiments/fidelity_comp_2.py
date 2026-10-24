# with this, the current experiment file is executable while being able to use src as a package
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cycle_shifts import CST, Domino, MaxSmallest
from src.mytools import CustomNoiseModel
from numpy import arange
import matplotlib.pyplot as plt
    

if __name__ == '__main__':
    """
    Calculates and plots the median evolution of the fidelity per width, per selected noisy backend.
    Compares all `CST` models in order of definition.
    """
    results = CustomNoiseModel.calculate_fidelities(
        CST.methods(k=4),
        max_width=10,
        date='7/7/26'
    )
    fig, axes = plt.subplots(nrows=1, ncols=len(results), figsize=(10, 3.5))
    maxsmall = MaxSmallest()
    domino = Domino()
    
    for ax, (backend, digest) in zip(axes, results.items()):
        best = 0
        for (label, report), marker in zip(digest.items(), 'osx*'):
            w = arange(1, len(report.improvements) + 1)
            report.plot_median_curve(ax, w, label=label.name, marker=marker)
            if label.name != domino.name: best = max(best, report.improvements[-1].median)
        
        for n, ms, dom in zip(w, digest[maxsmall].improvements, digest[domino].improvements):
            if ms.median > dom.median: print(f'{maxsmall.name} < {domino.name} for {n=} in {backend}')

        ax.set_yticks([best, report.improvements[-1].median, 1.0])
        ax.set_xticks(w)
        ax.set_title(backend)
        ax.legend()
    
    fig.suptitle('Median fidelity evolution per width')
    fig.supxlabel('Host register width n')
    plt.tight_layout()
    plt.show()
