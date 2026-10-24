# with this, the current experiment file is executable while being able to use src as a package
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cycle_shifts import CST
from src.mytools import sample
from time import perf_counter
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from contextlib import contextmanager
from numpy import median, mean
import matplotlib.pyplot as plt


@contextmanager
def garbage_collector (disable=False):
    """
    Gracefully handles GC on/off.
    """
    import gc

    if not disable: yield; return
    enabled = gc.isenabled()
    if enabled: gc.disable()
    try: yield 
    finally: 
        if enabled: gc.enable()


def time_all (width: int, max_samples: int, seed: int, magnitude_order=1000):
    """
    Times all `CST` methods on all sampled jumps. `magnitude_order=1000` (default) yields milliseconds.
    """
    samples = sample(width, max_samples, seed)
    median_times = defaultdict(float)

    for method in CST.methods():
        times = []
        for jump in samples:
            start = perf_counter()
            method.decompose(jump, width, as_tuple=True)  # the generator must be consumed but the result is irrelevant
            times.append((perf_counter() - start) * magnitude_order)

        median_times[method.name] = median(times)

    return width, median_times


def median_runtime (max_width: int, max_samples=10_000, seed=42, shots=10, disable_gc=False):
    """
    Calculates the median runtime of all `CST` methods, mean averaged over `shot` iterations.
    """
    with garbage_collector(disable_gc):
        # even with a fixed seed, results may vary between experiment shots due to cache misses etc. 
        # better to average across multiple shots
        tasks = [(n, max_samples, seed) for n in range(1, max_width + 1) for _ in range(shots)]
        with Pool(cpu_count()) as pool: results = pool.starmap(time_all, tasks)

    medians = defaultdict(lambda: defaultdict(list))
    for n, r in results:
        for label, median in r.items():
            medians[label][n].append(median)

    return {
        label: [mean(medians[label][n]) for n in range(1, max_width + 1)] 
        for label in medians
    }


if __name__ == '__main__':
    # NOTE: For max_width=50, #shots=10 and #cores=12, this takes ~20mins to run
    max_width = 50
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True)
    
    for i, ax in enumerate(axes):
        for m, (l, data) in zip('osx^', median_runtime(max_width, disable_gc=i).items(), strict=True):
            ax.plot(range(1, max_width + 1), data, label=l, marker=m, markevery=max(1, max_width // 10))
        
        ax.set_title('GC disabled' if i else 'GC enabled')
        ax.grid(True)
        ax.legend()

    fig.supxlabel('Host register width n')
    fig.supylabel('Median runtime (msec)')
    fig.suptitle('Median runtime comparison per register width')
    plt.tight_layout()
    plt.show()
