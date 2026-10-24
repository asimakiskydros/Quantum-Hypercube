from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error, ReadoutError
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import state_fidelity
from typing import Sequence, Iterable
from dataclasses import dataclass, field
from textwrap import wrap
from math import log2, ceil
from pandas import cut
import random
import numpy as np
import pandas as pd


def argfind (element, arr: Sequence):
    """
    `list.index(element)` but instead of `==` it compares by identity (`is`).
    """
    for i, x in enumerate(arr):
        if x is element: return i


def sign_magnitude (number: int, bits: int): 
    """
    Returns the sign of `number` and its magnitude scaled to fit in the given amount of `bits`.
    """
    return 1 if number > 0 else -1 if number < 0 else 0, abs(number) % (1 << bits)


def sample (width: int, ceiling: int, seed=42):
    """
    Samples values from the range [1, 2^width-1], with a limit of `ceiling` increased linearly according to width.
    """
    random.seed(seed)
    scale = width - ceil(log2(ceiling))  # guess the width in which the actual jump count nears the ceiling
    if scale < 0:  # so long as this is negative, the actual jump count is below the ceiling
        return list(range(1, 2 ** width))
    else:
        # scale is an index that says how many numbers `width` is after the pivot in which ceiling is preferable
        # because the true jump count grows exponentially, the sampled count should also grow (but linearly)
        return random.sample(range(1, 2 ** width), ceiling + scale * ceiling // 2)


def birange (low: int, high: int, step: int=1, reverse: bool=False):
    """
    Bi-directional range. 
    * `reverse=False` => `low`->`high` by `step`. 
    * `reverse=True` => `high`->`low` by `-step`.
    """
    start, stop, by = (high - 1, low - 1, -step) if reverse else (low, high, step)
    for i in range(start, stop, by): yield i


def as_coordinates (states: Iterable[str], width: int, dimensions: int):
    """
    Maps the basis states into the points they represent as integer tuples.
    """
    chunk = (width - 1) // dimensions

    for state in states: 
        yield tuple(int(dim, 2) for dim in wrap(state, chunk))


def is_power_of_two (number: int):
    """
    Returns `True` if `number` is a power of two.
    """
    return number > 0 and (number & (number - 1)) == 0
        

def minmedmax (array: Sequence, default=.0, as_dict=False):
    """
    Reduces the given `array` to its `min`, `median` and `max`, in that order.

    If `array` is empty it returns `default` on all of them.
    """
    _min = _med = _max = default

    if len(array) > 0:
        _min = min(array)
        _med = np.median(array)
        _max = max(array)
    
    if as_dict:
        return { 'min': _min, 'med': _med, 'max': _max }

    return _min, _med, _max


class Stats:
    """
    Collection of the relevant statistical data arising from the passed dataset (assumed applicable).

    All metrics skip potential `NaN`s.    
    """
    __slots__ = ('mean', 'median', 'perc25', 'perc75')

    def __init__(self, dataset: Sequence):
        ds = np.array(dataset)

        if ds.size == 0 or np.all(np.isnan(ds)):
            # this is the default behavior but this way silences runtime warnings
            self.mean = self.median = self.perc25 = self.perc75 = np.nan
        else:
            self.mean   = np.nanmean(ds)
            self.median = np.nanmedian(ds)
            self.perc25 = np.nanpercentile(ds, 25)
            self.perc75 = np.nanpercentile(ds, 75)


@dataclass(frozen=True, slots=True)
class Report:
    """
    Collection of experimental results.
    """
    label: str=''
    wins: list=field(default_factory=list)
    strict_wins: list= field(default_factory=list)
    improvements: list= field(default_factory=list)
    other: list=field(default_factory=list)


    def process (self, first, second, *, num_samples: int, metric=None):
        """
        Performs winrate/relative improvement processing on the passed arrays in their declared order
        and includes the resulting datapoints to their corresponding buckets.
        """
        metric = metric or (lambda a, b: (a - b) / a if a else np.nan)
        self.wins.append(sum(1 for f, s in zip(first, second, strict=True) if s <= f) * 100 / num_samples)
        self.strict_wins.append(sum(1 for f, s in zip(first, second, strict=True) if s < f) * 100 / num_samples)
        self.improvements.append(Stats([metric(*pub) for pub in zip(first, second, strict=True)]))


    def append (self, *, winrate=None, strict_winrate=None, improvement=None, other=None):
        """
        Convenience function to append one or multiple datapoints directly to the buckets.
        """
        if winrate is not None: self.wins.append(winrate)
        if strict_winrate is not None: self.strict_wins.append(strict_winrate)
        if improvement is not None: self.improvements.append(improvement)
        if other is not None: self.other.append(other)


    def plot_median_curve (self, medium, xticks, *, use_other=False, **kwargs):
        """
        Plots the median evolution curve of `improvements` on the artist `medium`.
        """
        data = self.other if use_other else self.improvements
        return medium.plot(xticks, [d.median for d in data], **kwargs, markevery=max(1, len(data) // 10))
        

    def plot_25_75_percentile_range (self, medium, xticks, h='///', c='lightskyblue', e='cornflowerblue', a=0.2, *, use_other=False, **kwargs):
        """
        Plots the 25th-75th percentile cloud of `improvements` on the artist `medium`.
        """
        data = self.other if use_other else self.improvements
        medium.fill_between(
            xticks, 
            [d.perc25 for d in data], 
            [d.perc75 for d in data],
            alpha=a, hatch=h, color=c, edgecolor=e, **kwargs)


    def add_winrates_on_legend (self, medium, hl=0, ht=0, fb=True, *, median_soft=False, median_strict=False):
        """
        Hijacks the legend to instead include the mean/median winrate(s) of the experiment.
        """
        soft_kind, soft_avg = ('Median', np.median) if median_soft else ('Mean', np.mean)
        strict_kind, strict_avg = ('Median', np.median) if median_strict else ('Mean', np.mean)
        medium.legend([
            f'{soft_kind} winrate: {soft_avg(self.wins):.2f}%\n'
            f'{strict_kind} strict winrate: {strict_avg(self.strict_wins):.2f}%'],
            handlelength=hl, handletextpad=ht, fancybox=fb)
        
    
    def median_bounds_on_yaxis (self, medium):
        """
        Labels the y-axis of `medium` on the `min` and `median` value of the median improvement recorded in the experiment.
        """
        bounds = minmedmax([d.median for d in self.improvements])[:-1]
        medium.set_yticks(bounds)
        medium.set_yticklabels([f'{y * 100:.2f}%' for y in bounds])


def cost_models (max_width: int):
    """
    Yields a dictionary of principal cost models for MCX decompositions.
    """
    # precomputes all possible triangle costs
    logs = np.log2(np.arange(1, max_width+1))
    cumulative = np.cumsum(logs ** 3)

    return {
        'M(m)=O(m)': lambda x: x * (x + 1) // 2,
        'M(m)=O(m^2)': lambda x: x * (x + 1) * (2 * x + 1) // 6,
        'M(m)=O(log^3m)': lambda x: cumulative[x - 1]  # equivalent to:  sum(log2(i) ** 3 for i in range(1, x + 1)), 
    }


def latexify (label: str):
    """
    Wraps the `label` in LaTeX inline-equation formatters, typesets `log`s and beautifies special characters ("M", "O").
    """
    return f'${label.replace('M', '\\mathcal{M}').replace('O', '\\mathcal{O}').replace('log', '\\log')}$'


def density_matrix (circuits, sim: AerSimulator, **kwargs):
    """
    Calculates the density matrices of the passed `circuit`s via `sim`.
    """
    res = sim.run(circuits, **kwargs).result()
    return [res.data(i)['density_matrix'] for i in range(len(res.results))]


def load_ibm_qpu_stats (date: str):
    """
    Yields a `pandas.DataFrame` compiling the error stats of the 10 active IBM QPUs.

    Taken from https://quantum.cloud.ibm.com/computers
    """
    match date:
        case '7/7/26':
            return pd.DataFrame([
                {"l": "ibm_boston",     "1qem": 1.530e-4, "2qem": 1.25e-3, "rem": 4.15e-3,  "t1m": 235.30e-6, "t2m": 275.24e-6, "tu1": 32e-9, "tu2": 68e-9 },
                {"l": "ibm_kingston",   "1qem": 2.371e-4, "2qem": 1.90e-3, "rem": 8.057e-3, "t1m": 247.03e-6, "t2m": 141.44e-6, "tu1": 32e-9, "tu2": 68e-9 },
                {"l": "ibm_pittsburgh", "1qem": 1.891e-4, "2qem": 1.59e-3, "rem": 4.272e-3, "t1m": 270.98e-6, "t2m": 306.27e-6, "tu1": 32e-9, "tu2": 88e-9 },
                {"l": "ibm_fez",        "1qem": 3.015e-4, "2qem": 2.81e-3, "rem": 8.972e-3, "t1m": 135.78e-6, "t2m": 98.980e-6, "tu1": 24e-9, "tu2": 68e-9 },
                {"l": "ibm_marrakesh",  "1qem": 3.568e-4, "2qem": 2.69e-3, "rem": 1.257e-2, "t1m": 153.53e-6, "t2m": 83.830e-6, "tu1": 36e-9, "tu2": 68e-9 },
                {"l": "ibm_miami",      "1qem": 3.676e-4, "2qem": 3.07e-3, "rem": 2.466e-2, "t1m": 319.50e-6, "t2m": 217.29e-6, "tu1": 32e-9, "tu2": 128e-9},
                {"l": "ibm_aachen",     "1qem": 2.226e-4, "2qem": 1.65e-3, "rem": 4.761e-3, "t1m": 213.88e-6, "t2m": 254.01e-6, "tu1": 32e-9, "tu2": 68e-9 },
                {"l": "ibm_berlin",     "1qem": 1.644e-4, "2qem": 2.09e-3, "rem": 2.106e-2, "t1m": 327.10e-6, "t2m": 251.24e-6, "tu1": 32e-9, "tu2": 68e-9 },
            ])
        case '21/4/26':
            return pd.DataFrame([
                {"l": "ibm_boston",     "1qem": 1.956e-4, "2qem": 1.19e-3, "rem": 5.13e-3, "t1m": 281.67e-6, "t2m": 326.63e-6, "tu1": 32e-9, "tu2": 68e-9 },
                {"l": "ibm_kingston",   "1qem": 2.439e-4, "2qem": 1.87e-3, "rem": 9.16e-3, "t1m": 273.16e-6, "t2m": 135.63e-6, "tu1": 32e-9, "tu2": 68e-9 },
                {"l": "ibm_pittsburgh", "1qem": 1.925e-4, "2qem": 1.67e-3, "rem": 4.52e-3, "t1m": 295.57e-6, "t2m": 335.77e-6, "tu1": 32e-9, "tu2": 88e-9 },
                {"l": "ibm_fez",        "1qem": 2.883e-4, "2qem": 2.68e-3, "rem": 1.48e-2, "t1m": 141.28e-6, "t2m": 103.45e-6, "tu1": 24e-9, "tu2": 68e-9 },
                {"l": "ibm_marrakesh",  "1qem": 3.27e-4,  "2qem": 2.70e-3, "rem": 1.12e-2, "t1m": 187.57e-6, "t2m": 105.69e-6, "tu1": 36e-9, "tu2": 68e-9 },
                {"l": "ibm_miami",      "1qem": 2.137e-4, "2qem": 2.56e-3, "rem": 1.89e-2, "t1m": 345.89e-6, "t2m": 265.66e-6, "tu1": 32e-9, "tu2": 128e-9},
                {"l": "ibm_aachen",     "1qem": 1.844e-4, "2qem": 1.56e-3, "rem": 6.59e-3, "t1m": 238.42e-6, "t2m": 265.65e-6, "tu1": 32e-9, "tu2": 68e-9 },
                {"l": "ibm_brussels",   "1qem": 2.637e-4, "2qem": 8.99e-3, "rem": 2.89e-2, "t1m": 232.67e-6, "t2m": 128.33e-6, "tu1": 60e-9, "tu2": 660e-9},
                {"l": "ibm_strasbourg", "1qem": 2.545e-4, "2qem": 7.82e-3, "rem": 2.60e-2, "t1m": 265.33e-6, "t2m": 103.75e-6, "tu1": 60e-9, "tu2": 660e-9},
                {"l": "ibm_berlin",     "1qem": 1.673e-4, "2qem": 2.26e-3, "rem": 2.14e-2, "t1m": 344.72e-6, "t2m": 248.65e-6, "tu1": 32e-9, "tu2": 68e-9 },
            ])
        case '4/3/26':
            return pd.DataFrame([
                {"l": "ibm_boston",     "1qem": 1.627e-4, "2qem": 1.30e-3,  "rem": 4.88e-3, "t1m": 279.48e-6, "t2m": 339.77e-6, "tu1": 32e-9, "tu2": 68e-9 },
                {"l": "ibm_kingston",   "1qem": 2.043e-4, "2qem": 2.03e-3,  "rem": 1.67e-2, "t1m": 248.54e-6, "t2m": 139.81e-6, "tu1": 32e-9, "tu2": 68e-9 },
                {"l": "ibm_pittsburgh", "1qem": 1.784e-4, "2qem": 1.58e-3,  "rem": 1.26e-2, "t1m": 303.54e-6, "t2m": 327.38e-6, "tu1": 32e-9, "tu2": 88e-9 },
                {"l": "ibm_fez",        "1qem": 3.135e-4, "2qem": 2.64e-3,  "rem": 1.33e-2, "t1m": 143.29e-6, "t2m": 98.24e-6,  "tu1": 24e-9, "tu2": 68e-9 },
                {"l": "ibm_marrakesh",  "1qem": 2.922e-4, "2qem": 2.47e-3,  "rem": 1.17e-2, "t1m": 196.11e-6, "t2m": 103.3e-6,  "tu1": 36e-9, "tu2": 68e-9 },
                {"l": "ibm_torino",     "1qem": 3.008e-4, "2qem": 2.46e-3,  "rem": 4.81e-2, "t1m": 171.76e-6, "t2m": 132.32e-6, "tu1": 32e-9, "tu2": 68e-9 },
                {"l": "ibm_miami",      "1qem": 1.938e-4, "2qem": 2.71e-3,  "rem": 1.92e-2, "t1m": 352.24e-6, "t2m": 265.42e-6, "tu1": 32e-9, "tu2": 128e-9},
                {"l": "ibm_aachen",     "1qem": 1.995e-4, "2qem": 1.83e-3,  "rem": 4.15e-3, "t1m": 225.32e-6, "t2m": 252.98e-6, "tu1": 32e-9, "tu2": 68e-9 },
                {"l": "ibm_brussels",   "1qem": 2.574e-4, "2qem": 7.744e-3, "rem": 3.38e-2, "t1m": 239.48e-6, "t2m": 147.11e-6, "tu1": 60e-9, "tu2": 660e-9},
                {"l": "ibm_strasbourg", "1qem": 2.573e-4, "2qem": 8.04e-3,  "rem": 2.92e-2, "t1m": 262.41e-6, "t2m": 119.09e-6, "tu1": 60e-9, "tu2": 660e-9}
            ])
        case _:
            raise ValueError(f'Unsupported `date` passed in `load_ibm_qpu_stats() (received {date})')


class CustomNoiseModel(NoiseModel):
    def __init__(self, noise, q1=('x', 'sx', 'rz', 'id'), q2=('cx', 'cz'), basis_gates=None):
        """
        Builds a custom idealized `NoiseModel` backend based on the `noise` data decided. Includes one and two qubit gate error,
        as well as readout error. Considers the instruction set `X, sqrt(X), RZ, I, CX, CZ` by default.
        """
        super().__init__(basis_gates)
        tpattern = lambda tu: (noise['t1m'], noise['t2m'], noise[tu])
        treU2 = thermal_relaxation_error(*tpattern('tu2'))
        rem = noise['rem']
        # single qubit gates
        self.add_all_qubit_quantum_error(depolarizing_error(noise['1qem'], 1).compose(thermal_relaxation_error(*tpattern('tu1'))), q1)
        # two qubit gates
        self.add_all_qubit_quantum_error(depolarizing_error(noise['2qem'], 2).compose(treU2.tensor(treU2)), q2)
        # readout error
        self.add_all_qubit_readout_error(ReadoutError([[1 - rem, rem], [rem, 1 - rem]]))
        self.name = f'Fake{noise["l"].split("_")[1].title()}'


    @classmethod
    def backends (cls, date: str=None, segregator='2qem', bins=(0, 2e-3, 3e-3, float('inf')), group_name='noise_group', seed=42):
        """
        Loads idealized QPU backend stats, segregates them (by default by two-gate error) into three noise groups,
        low, mid, high, then samples one candidate from each group.
        """
        df = load_ibm_qpu_stats(date).sort_values(by=segregator)
        df[group_name] = cut(df[segregator], bins=bins)
        selections = df.groupby(group_name, observed=False).apply(lambda x: x.sample(1, random_state=seed), include_groups=False)
        return [CustomNoiseModel(df.loc[df['l'] == selection].iloc[0]) for selection in selections['l']]
    

    @classmethod
    def calculate_fidelities (cls, models: Iterable, *, max_width: int, date: str, max_samples: int=10, seed=42):
        """
        Calculates median fidelity evolution on all `models`, for samples up until `max_width`, on IBM data retreived at `date`.

        NOTE: with `max_width=10` this takes at least an hour.
        """
        results: dict[str, dict[str, Report]] = {}
        all_samples = [sample(n, max_samples, seed) for n in range(1, max_width + 1)]

        for backend in CustomNoiseModel.backends(date):
            results[backend.name] = { 
                method: backend.fidelity(all_samples, backend, method) for method in models
            }

        return results


    def fidelity (self, all_samples, noise_backend, cst, verbose=True):
        """
        Compares measurement accuracy between ideal `Aer` and a version litered by `noise_backend` by fidelity.
        """
        ideal = AerSimulator(method='density_matrix')
        noisy = AerSimulator(method='density_matrix', noise_model=noise_backend)

        report = Report()
        for n, samples in enumerate(all_samples):
            circuits = []
            for jump in samples:
                circuit = QuantumCircuit(reg := QuantumRegister(n + 1))
                circuit = cst.shift(by=jump, on=reg, circuit=circuit)
                circuit.save_density_matrix()
                circuits.append(circuit)

            isos = transpile(circuits, ideal)
            dm_ideal = density_matrix(isos, ideal, validate=False)
            dm_noisy = density_matrix(isos, noisy, validate=False)
            report.improvements.append(Stats([state_fidelity(*pub) for pub in zip(dm_ideal, dm_noisy, strict=True)]))

        if verbose: print(f'Done with {cst} on {noise_backend.name}')

        return report

