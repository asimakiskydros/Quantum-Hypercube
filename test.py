from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from typing import Callable, Literal
from itertools import product
from math import log2
from textwrap import wrap
import matplotlib.pyplot as plt


def load_shape (dimensions: int, size: int, rule: Callable, show = False):
    """
    Maps the given rule into a quantum binary image of given `dimensions` and given `size` per dimension.
    """
    # ensure `size` is a power of two
    assert size > 0 and (size & (size - 1)) == 0

    qubits = int(log2(size))
    pos = [QuantumRegister(qubits, name=f'p{i}') for i in range(dimensions)]
    col = QuantumRegister(1, name='clr')
    circuit = QuantumCircuit(*[*pos, col])
    # the qubits of all positional registers unwrapped for easy mcx declarations
    controls = [q for p_reg in pos for q in p_reg] 

    # initialize indexers
    for p_reg in pos: 
        circuit.h(p_reg)

    # generate every possible point and if `rule` accepts it, add a MCX gate with controls 
    # according to its coordinates, targetting `col`.
    for point in product(range(size), repeat=dimensions):
        if rule(*point):
            control_state = ''.join([bin(coordinate)[2:].zfill(qubits) for coordinate in reversed(point)])
            circuit.mcx(controls, col, ctrl_state=control_state)

    if show:
        circuit.draw('mpl')
        plt.show()
    
    return circuit, pos, col


def probability_vector (circuit: QuantumCircuit, shots = 10_000, condition: Literal['0', '1'] = '1'):
    """
    Measures all qubits and returns the quasi-probability vector calculated from the resulting counts.
    """
    # NOTE: My dumbass thought that I could use Statevector(...) instead of this entire routine 
    # HAHA! v1.3.1 it yields garbage for size=4 and size=8, v.1.2.4 yields garbage only for size=8
    # This works on all versions for all sizes. 
    circuit.measure_all()
    
    backend = AerSimulator()
    counts: dict[str, int] = backend.run(transpile(circuit, backend), shots=shots).result().get_counts()
    counts = { state[1:]: hits for state, hits in counts.items() if state[0] == condition }
    norm = sum(hits for hits in counts.values())

    circuit.remove_final_measurements()

    return { state: hits / norm for state, hits in counts.items() }


def plot (dimensions: int, circuit: QuantumCircuit, show = False, highlight: tuple[str] = None):
    """
    Plots the given circuit in the asked dimensions.
    """
    if dimensions not in (2, 3): 
        raise NotImplementedError(f'Cannot plot {dimensions}-dimensional plot yet.')
    
    probs = probability_vector(circuit)

    # split basis states back into the coordinates they represent
    # and add them to the appropriate dimensional sublist
    coords = [[] for _ in range(dimensions)]
    hcoords = None
    for state in probs.keys():
        point = [int(dim, 2) for dim in wrap(state, (circuit.num_qubits - 1) // dimensions)]
        if highlight is not None and point == highlight:
            hcoords = point
            continue

        for sublist, dim in zip(coords, point, strict=True):
            sublist.append(dim)

    if dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots()

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if dimensions == 3:
        ax.set_zlabel('z')

    ax.scatter(*reversed(coords), c='blue')
    
    if hcoords:
        ax.scatter(*reversed(hcoords), c='red', marker='x')
        ax.text(
            *(c + 0.2 for c in reversed(hcoords)), 
            f'({", ".join(str(c) for c in reversed(hcoords))})', color='red')
    
    if show:
        plt.show()

    return fig


def cycle_shift (circuit: QuantumCircuit, pos: QuantumRegister, step: int = 1, show = False, barrier = True):
    """
    Shifts mod 2**len(pos) along the dimension expressed by `pos`, for the given amount of `step`s.
    """
    if barrier:
        circuit.barrier()

    # can only truly shift by mod 2**n jumps, so normalize the step to this region
    # if the true jump is 0 steps, do nothing
    if (_step := abs(step) % 2 ** pos.size) == 0: 
        if show:
            circuit.draw('mpl')
            plt.show()

        return
    # work recursively to save gates; jump by the largest power of 2 thats less than the remaining step
    largest_p2 = 1 << (_step.bit_length() - 1)
    # to jump by a power of 2, start the shifting log(2**k) qubits later
    padding = int(log2(largest_p2))
    # reflect the circuit structure if the step is negative
    adjust = lambda x: (pos.size - 1 - x if step > 0 else x)

    for controls, qubit in enumerate(reversed(pos) if step > 0 else pos):
        if pos.index(qubit) < padding: continue  # skip the padding qubits
        
        controls = adjust(controls)

        if controls == padding: circuit.x(qubit)  # MCX doesn't generalize to 0 controls, add the lone X gate manually
        else: circuit.mcx(pos[padding:controls], qubit)
    # perform a cycle shift for the remaining step amount
    cycle_shift(circuit, pos, (_step - largest_p2) * (1 if step > 0 else -1), barrier=False)


if __name__ == '__main__':
    dims = 3
    size = 8

    # TESTS: draw all possible diagonals 
    # rule = lambda x, y: x == y or x == size - 1 - y  # works only for dims = 2
    rule = lambda x, y, z: (  # works only for dims = 3
        x == y == z or 
        x == y == size - 1 - z or 
        x == size - 1 - y == z or
        x == size - 1 - y == size - 1 - z
        )

    qc, p, c = load_shape(dims, size, rule, show=False)
    fig1 = plot(dims, qc, highlight=[1, 1, 1], show=False)

    cycle_shift(qc, p[0], 3) # move 3 steps along the x-axis
    cycle_shift(qc, p[1], 2) # move 2 steps along the y-axis
    cycle_shift(qc, p[2], 1) # move 1 steps along the z-axis
    
    # coords are read in reverse order: [(z, ) y, x]
    fig2 = plot(dims, qc, highlight=[2, 3, 4], show=False)

    qc.draw('mpl')
    plt.show()