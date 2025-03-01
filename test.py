from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from typing import Callable, Literal, Sequence, Mapping
from itertools import product
from math import log2
from textwrap import wrap
import matplotlib.pyplot as plt


def encoding (dimensions: int, size: int, *rules: list[Callable]):
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
    
    return circuit, pos, colors


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
    # # reflect the circuit structure if the step is negative
    start, stop, by = (pos.size - 1, padding - 1, -1) if step > 0 else (padding, pos.size, 1)
    for i in range(start, stop, by):
        if i == padding:
            circuit.x(pos[i])  # qiskit doesnt support declaration of a mcx with 0 controls through .mcx
        else:
            circuit.mcx(pos[padding:i], pos[i])

    # perform a cycle shift for the remaining step amount
    cycle_shift(circuit, pos, (_step - largest_p2) * (1 if step > 0 else -1), barrier=False)


class Plotter:
    """
    Custom context manager to handle shifting and plotting more clearly
    """
    def __init__(self, circuit: QuantumCircuit, axes: Sequence[QuantumRegister], colors: Sequence[int]):
        if (dims := len(axes)) not in (2, 3): raise NotImplementedError(f'Cannot plot {dims}D structures.')

        self.circ = circuit
        self.axes = axes
        self.a_colors = colors  # og image color/point
        self.p_colors: dict[tuple[int], int] = self.a_colors.copy()  # shifted image color/point
        self.apriori = statevector(self.circ)  # keep a record of the original image
    
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

        for state in states: yield tuple(int(dim, 2) for dim in wrap(state, chunk))

    def shift (self, axis: Sequence[int], step: int = 1):
        """
        Convenience function to be used inside the context manager. Allows declaring the entire shift-axis at once
        and how many steps along its path to take.
        """
        for dim, scalar in enumerate(axis): cycle_shift(self.circ, self.axes[dim], step * scalar)

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


def test_2d (size: int):
    """
    Generates the two diagonals of a square of side `size`.
    """
    return encoding(2, size, 
        lambda x, y: x == y, 
        lambda x, y: x == size - 1 - y
        )

def test_3d (size: int):
    """
    Generates the four diagonals of a cube of side `size`.
    """
    return encoding(3, size, 
        lambda x, y, z: x == y == z, 
        lambda x, y, z: x == y == size - 1 - z,
        lambda x, y, z: x == size - 1 - y == z,
        lambda x, y, z: x == size - 1 - y == size - 1 - z
        )

if __name__ == '__main__':
    qc, p, c = test_2d(8)
    # qc, p, c = test_3d(8)
    
    with Plotter(qc, p, c) as pltr:
        # move one step along the x/3=y/2=z axis
        # ie mean 3 steps along the x-axis, 2 along the y-axis and 1 along the z-axis
        # pltr.shift([3, 2, 1], step=1)  # NOTE: this obviously only works for the 3D test. Define some other shift for 2D.
        pltr.shift([3, 2], step=3)
