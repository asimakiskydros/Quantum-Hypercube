from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from typing import Callable, Literal, Sequence, Mapping
from itertools import product
from math import log2
from textwrap import wrap
import matplotlib.pyplot as plt


def encoding (dimensions: int, size: int, rule: Callable):
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
    
    return circuit, pos


def statevector (circuit: QuantumCircuit, shots = 10_000, condition: Literal['0', '1'] = '1') -> list[str]:
    """
    Measures all qubits and returns the quasi-probability vector calculated from the resulting counts.
    """
    circuit.measure_all()
    
    backend = AerSimulator()
    counts: Mapping[str, int] = backend.run(transpile(circuit, backend), shots=shots).result().get_counts()

    circuit.remove_final_measurements()

    return [state[1:] for state in counts.keys() if state[0] == condition]


def cycle_shift (circuit: QuantumCircuit, pos: QuantumRegister, step: int = 1, show = False, barrier = True):
    """
    Shifts mod 2**len(pos) along the dimension expressed by `pos`, for the given amount of `step`s.
    """
    if barrier: circuit.barrier()
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


class Plotter:
    """
    Custom context manager to handle shifting and plotting more clearly
    """
    def __init__(self, circuit: QuantumCircuit, axes: Sequence[QuantumRegister]):
        if (dims := len(axes)) not in (2, 3):
            raise NotImplementedError(f'Cannot plot {dims}D structures.')

        self.circ = circuit
        self.axes = axes
        self.apriori = statevector(self.circ)  # keep a record of the original image
        self.a_colors = { state: i for i, state in enumerate(self.explode(self.apriori)) }  # og image color/point
        self.p_colors: dict[tuple[int], int] = self.a_colors.copy()  # shifted image color/point
        
    def __enter__(self):
        return self

    def __exit__(self, *args):
        posteriori = statevector(self.circ)  # snapshot the shifted image
        self.plot(self.apriori, self.a_colors, 'Before')  # plot original
        self.plot(posteriori, self.p_colors, 'After')  # plot shifted
        self.circ.draw('mpl')  # plot circuit
        plt.show()

    def explode (self, states: Sequence[str]):
        """
        Maps the basis states into the points they represent as integer tuples.
        """
        chunk = (self.circ.num_qubits - 1) // len(self.axes)

        for state in states:
            yield tuple(int(dim, 2) for dim in wrap(state, chunk))

    def shift (self, axis: Sequence[int], step: int = 1):
        """
        Convenience function to be used inside the context manager. Allows declaring the entire shift-axis at once
        and how many steps along its path to take.
        """
        for dim, scalar in enumerate(axis):
            cycle_shift(self.circ, self.axes[dim], step * scalar)

        updated = self.p_colors.copy()
        overflow = 1 << self.axes[0].size

        for point, value in self.p_colors.items():
            # update the shifted color/point record by updating the keys but keeping the colors the same
            new_point = tuple((pi + x * step) % overflow for pi, x in zip(point, reversed(axis), strict=True))
            updated[new_point] = value
        
        self.p_colors = updated

    def plot (self, dataset: Sequence[str], colors: Mapping[tuple[int], int], title: str):
        """
        Plots the given dataset using the given colors.
        """
        if (dims := len(self.axes)) == 2: fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlabel('z')

        coords, cols = [[] for _ in range(dims)], []
        for point in self.explode(dataset):
            cols.append(colors[point])

            for sublist, pi in zip(coords, point, strict=True):
                sublist.append(pi)

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(*reversed(coords), c=cols)

        return fig


if __name__ == '__main__':
    dims2, dims3, size = 2, 3, 8

    # TESTS: draw all possible diagonals 
    rule2 = lambda x, y: x == y or x == size - 1 - y  # works only for dims = 2
    rule3 = lambda x, y, z: (  # works only for dims = 3
        x == y == z or 
        x == y == size - 1 - z or 
        x == size - 1 - y == z or
        x == size - 1 - y == size - 1 - z
        )

    # qc, p = encoding(dims2, size, rule2)
    qc, p = encoding(dims3, size, rule3)
    
    with Plotter(qc, p) as pltr:
        # move one step along the x/3=y/2=z axis
        # ie mean 3 steps along the x-axis, 2 along the y-axis and 1 along the z-axis
        pltr.shift([3, 2, 1], step=1)
        