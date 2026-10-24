"""
Multi-dimensional cycle shift example.
Relevant to: https://doi.org/10.1088/1402-4896/ae1d2b
"""

# with this, the current experiment file is executable while being able to use src as a package
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from typing import Sequence, Callable, Mapping, Iterable
from src.mytools import as_coordinates
from src.cycle_shifts import MaxSmallest, CST
from dataclasses import dataclass
from itertools import product
from contextlib import contextmanager
import matplotlib.pyplot as plt


def encoding (dimensions: int, size: int, *rules: list[Callable], encapsulate=True):
    """
    Maps the given rule into a quantum binary image of given `dimensions` and given `size` per dimension.
    """
    # ensure `size` is a power of two
    assert size > 0 and (size & (size - 1)) == 0

    qubits = size.bit_length() - 1
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
        circuit.append(gate, [*(q for reg in pos for q in reg), col])

    return circuit, pos, colors


def experiment (*, side: int, d: int):
    """
    Generates the diagonals of a hypercube of side `side` and dimensions `d`.
    """
    match d:
        case 2:
            return encoding(2, side, 
                lambda x, y: x == y, 
                lambda x, y: x == side - 1 - y)
        case 3:
            return encoding(3, side, 
                lambda x, y, z: x == y == z, 
                lambda x, y, z: x == y == side - 1 - z,
                lambda x, y, z: x == side - 1 - y == z,
                lambda x, y, z: x == side - 1 - y == side - 1 - z)        
        case _:
            raise NotImplementedError(f'Experiments for dimensions not in (2, 3) are not supported! (Received {d})')


@contextmanager
def temporary_measurements(circuit: QuantumCircuit):
    """
    Gracefully appends and removes measurements to all qubits.
    """
    circuit.measure_all()
    try:
        yield circuit
    finally:
        circuit.remove_final_measurements()


@dataclass
class Plotter:
    """
    Custom context manager to handle shifting and plotting more clearly.
    """
    circuit: QuantumCircuit
    axes: Sequence[QuantumRegister]
    colors: Mapping
    method: CST
    new_colors: Mapping=None


    def __enter__(self):
        self.apriori = self.approximate_image()  # keep a record of the original image
        return self


    def __exit__(self, *_):
        posteriori = self.approximate_image()  # snapshot the shifted image
        self.plot(self.apriori, self.colors, f'{self.method.name}: Before')  # plot original
        self.plot(posteriori, self.new_colors, f'{self.method.name}: After')     # plot shifted
        self.circuit.draw('mpl')  # plot circuit
        # plt.title(f'Method: {self.method.name} power of two')
        plt.show()


    def approximate_image (self, shots=10_000):
        """
        Measures all qubits and approximates the image from the quasi-probability vector.
        """
        with temporary_measurements(self.circuit) as circ:
            sim = AerSimulator()
            counts = sim.run(circ, shots=shots).result().get_counts()
            return [state[1:] for state in counts if state[0] == '1']


    def shift (self, axis: Sequence[int], step: int=1):
        """
        Convenience function to be used inside the context manager. Allows declaring the entire shift-axis at once
        and how many steps along its path to take.
        """
        self.method.shift(by=step, on=self.axes, circuit=self.circuit, shift_axis=axis)
        self.new_colors = self.colors.copy()
        overflow = 1 << self.axes[0].size

        for point, value in self.colors.items():
            # update the shifted color/point record by updating the keys but keeping the colors the same
            new_point = tuple((pi + x * step) % overflow for pi, x in zip(point, reversed(axis), strict=True))
            self.new_colors[new_point] = value


    def plot (self, dataset: Iterable[str], colors: Mapping[tuple[int], int], title: str):
        """
        Plots the given dataset using the given colors.
        """
        if (dims := len(self.axes)) not in (2, 3): raise NotImplementedError(f'Cannot plot {dims}D structures.')
        if dims == 2: fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlabel('z')

        coords, cols = [[] for _ in range(dims)], []
        for point in as_coordinates(dataset, self.circuit.num_qubits, len(self.axes)):
            cols.append(colors[point])
            for sublist, pi in zip(coords, point, strict=True): sublist.append(pi)

        # ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(*reversed(coords), c=cols)
        return fig


if __name__ == '__main__':
    with Plotter(*experiment(side=8, d=2), MaxSmallest(barriers=True)) as pltr:
        """ 
        []======================================= 2D EXAMPLE =======================================[]
          Move three steps along the x/3=y/2 axis.
          This implies 3x3=9 steps along the x-axis and 3x2=6 steps along the y-axis;
          however, since 9 mod 8 = 1 (8 is the image side size), the x-axis step will actually be 1.
        """
        pltr.shift([3, 2], step=3)

    with Plotter(*experiment(side=8, d=3), MaxSmallest(barriers=True)) as pltr:
        """ 
        []======================================= 3D EXAMPLE =======================================[]
          Move one step along the x/3=y/2=z axis.
          This implies 1x3=3 steps along the x-axis, 1x2=2 along the y-axis and 1x1=1 along the z-axis.
        """
        pltr.shift([3, 2, 1], step=1)
        