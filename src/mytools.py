from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from typing import Callable, Mapping
from itertools import product
from math import log2, ceil
import matplotlib.pyplot as plt
import random
import numpy as np


def encoding (dimensions: int, size: int, *rules: list[Callable], encapsulate=True):
    """
    Maps the given rule into a quantum binary image of given `dimensions` and given `size` per dimension.
    """
    # ensure `size` is a power of two
    assert size > 0 and (size & (size - 1)) == 0

    qubits = int(log2(size))
    pos = [QuantumRegister(qubits, name=f'p{i}') for i in range(dimensions)]
    col = QuantumRegister(1, name='clr')
    circuit = QuantumCircuit(*pos, col)
    # the qubits of all positional registers unwrapped for easy mcx declarations
    controls = [q for p_reg in pos for q in p_reg]
    # the color of each point
    colors: dict[tuple[int], int] = {}

    # initialize indexers
    for p_reg in pos: circuit.h(p_reg)
    # generate every possible point and if any of the `rules` accepts it, add a MCX gate with controls 
    # according to its coordinates, targetting `col`.
    for point in product(range(size), repeat=dimensions):
        for color, rule in enumerate(rules):
            if rule(*point):
                control_state = ''.join([bin(coordinate)[2:].zfill(qubits) for coordinate in reversed(point)])
                circuit.mcx(controls, col, ctrl_state=control_state)
                colors[point] = color  # paint the pixel according to the rule that accepted it
                break
    
    if encapsulate:
        # if desired, hide the implementation inside a black box
        gate = circuit.to_gate(label='Encoding')
        circuit.data.clear()
        circuit.append(gate, [q for reg in pos for q in reg] + [col])

    return circuit, pos, colors


def statevector (circuit: QuantumCircuit, shots=10_000, postselect: int=1) -> list[str]:
    """
    Measures all qubits and returns the quasi-probability vector calculated from the resulting counts.

    `postselect` > 0 means filtering counts by MSB = |1>, < 0 means filtering by MSB = |0> and 0 means no filter.
    """
    circuit.measure_all()
    backend = AerSimulator()
    counts: Mapping[str, int] = backend.run(transpile(circuit, backend), shots=shots).result().get_counts()
    condition = lambda x: x == '1' if postselect > 0 else x == '0' if postselect < 0 else True
    circuit.remove_final_measurements()
    return [state[1:] for state in counts.keys() if condition(state[0])]


def cycle_shift (circuit: QuantumCircuit, dimension: QuantumRegister, jump: int, forwards: bool=True):
    """
    Builds a quantum operator implementing a cycle shift by `jump` units, a power of two, `forwards` or backwards.
    """
    # only perform power-of-two jumps that are within the register's scope
    # since only powers of two are assumed, a jump of `dimension.size` or more is equal to 0 (mod 2^dim.size)
    if not is_power_of_two(jump) or jump >= (1 << dimension.size): return
    # to jump by a power of 2, start the shifting log(2**k) qubits later
    padding = int(log2(jump))
    # reflect the circuit structure if the step is negative
    start, stop, by = (dimension.size - 1, padding - 1, -1) if forwards else (padding, dimension.size, 1)
    for i in range(start, stop, by):
        if i == padding:
            circuit.x(dimension[i])  # qiskit doesnt support declaration of a mcx with 0 controls through .mcx
        else:
            circuit.mcx(dimension[padding:i], dimension[i])


def render (circuit: QuantumCircuit, method='mpl'):
    """
    Plots and immediately renders the circuit.
    """
    circuit.draw(method)
    plt.show()


def is_power_of_two (number: int):
    """
    Returns `True` if `number` is a power of two.
    """
    return number > 0 and (number & (number - 1)) == 0


def sgn_mag (number: int, bits: int): 
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
    

def stats (data: list):
    """
    Yields relevant statistical data arising from the given dataset (assumed applicable): 
    `mean`, `std`, `median`, `25th percentile`, `75th percentile`.

    All metrics skip potential `NaN`s.
    """
    arr = np.array(data, dtype=float)
    return {
        'mean':   float(np.nanmean(arr)),
        'std':    float(np.nanstd(arr, ddof=0)),
        'median': float(np.nanmedian(arr)),
        'perc25': float(np.nanpercentile(arr, 25)),
        'perc75': float(np.nanpercentile(arr, 75))
    }