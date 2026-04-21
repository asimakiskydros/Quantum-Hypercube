"""
    Experiment #5: approximates real noisy hardware and tests a small sample of cycle shifts on them using all
    algorithms (Max Smallest, Nearest, DP-DFS).
"""

# with this, the current experiment file is executable while being able to use src as a package
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mytools import sample, stats
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error, ReadoutError
from src.base_cst import cst_max_smallest
from src.greedy_cst import cst_nearest
from src.dp_cst import cst_dfs
from functools import partial
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_ibm_qpu_stats ():
    """
    Yields a `pandas.DataFrame` compiling the error stats of the 10 active IBM QPUs.

    Taken from https://quantum.cloud.ibm.com/computers on April 21st 2026.
    """
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


class CustomNoiseModel(NoiseModel):
    def __init__(self, noise, q1=('x', 'sx', 'id'), q2=('cx', 'cz'), basis_gates=None):
        """
        Builds a custom idealized `NoiseModel` backend based on the `noise` data decided. Includes one and two qubit gate error,
        as well as readout error. Considers the instruction set `X, sqrt(X), I, CX, CZ` by default (also `RZ`, considered virtual).
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
    def backends (cls, segregator='2qem', bins=(0, 2e-3, 3e-3, float('inf')), group_name='noise_group', seed=42):
        """
        Loads idealized QPU backend stats, segregates them (by default by two-gate error) into three noise groups,
        low, mid, high, then samples one candidate from each group.
        """
        df = load_ibm_qpu_stats().sort_values(by=segregator)
        df[group_name] = pd.cut(df[segregator], bins=bins)
        selections = df.groupby(group_name, observed=False).apply(lambda x: x.sample(1, random_state=seed), include_groups=False)
        return [CustomNoiseModel(df.loc[df['l'] == selection].iloc[0]) for selection in selections['l']]
    

def compare (all_samples, noise_backend, cst):
    """
    Compares measurement accuracy between ideal `AerSimulator` and a version litered by `noise_backend` by fidelity.
    """
    ideal = AerSimulator(method='density_matrix')
    noisy = AerSimulator(method='density_matrix', noise_model=noise_backend)
    fid_stats = []

    for n, samples in enumerate(all_samples):
        reg =  QuantumRegister(n + 1)
        circuits = []
        for jump in samples:
            circuit = QuantumCircuit(reg)
            cst(circuit, reg, jump)
            circuit.save_density_matrix()
            circuits.append(circuit)
        
        isos = transpile(circuits, ideal)
        res_ideal = ideal.run(isos, validate=False).result()
        res_noisy = noisy.run(isos, validate=False).result()
        fid_stats.append(stats([
            state_fidelity(res_ideal.data(i)['density_matrix'], res_noisy.data(i)['density_matrix'])
            for i in range(len(res_ideal.results))]
        ))

    return fid_stats


def plot_fidelities (results):
    """
    Plots the median evolution of the fidelity per width.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 5))
    maxminmed = lambda arr: (min(arr), np.median(arr), max(arr))
    
    for ax_col, (backend, digest) in zip(axes, results.items()):
        for ax, (cst, report) in zip(ax_col, digest.items()):
            w = np.arange(1, len(report) + 1)
            ax.plot(w, [d['median'] for d in report], markevery=len(report) // 10)
            ax.fill_between(w, [d['perc25'] for d in report], [d['perc75'] for d in report],
                            alpha=0.2, hatch='///', color='lightskyblue', edgecolor='cornflowerblue')    
            bounds = maxminmed([d['median'] for d in report])
            ax.set_yticks(bounds[:-1])
            ax.set_yticklabels([f'{y * 100:.2f}%' for y in bounds[:-1]])
            if ax is ax_col[0]: ax.set_ylabel(backend)  # label rows according to noise backend used
            if ax in axes[0]: ax.set_title(cst)         # label columns according to algorithm used
    
    fig.suptitle('Median fidelity evolution per algorithm per width')
    fig.supxlabel('Host register width n')

    return fig


if __name__ == '__main__':
    max_width = max_samples = 10
    verbose = True

    all_samples = [sample(n, max_samples) for n in range(1, max_width + 1)]
    results = defaultdict(defaultdict)
    for backend in CustomNoiseModel.backends():
        for idx, method in [
            ('1', cst_max_smallest), 
            ('2', cst_nearest), 
            ('4*', partial(cst_dfs, f=lambda x:x))
            ]: 
            results[backend.name][f'Alg. {idx}'] = compare(all_samples, backend, method)
            if verbose: print(f'Done with Alg. {idx} on {backend.name}')

    f = plot_fidelities(results)
    plt.tight_layout()
    plt.show()