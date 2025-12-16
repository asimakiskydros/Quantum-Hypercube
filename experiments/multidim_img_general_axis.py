"""
    Multi-dimensional cycle shift example.
    Relevant to: https://doi.org/10.1088/1402-4896/ae1d2b
"""


from qiskit import QuantumCircuit, QuantumRegister
from typing import Sequence, Callable, Mapping
from textwrap import wrap
from src.mytools import encoding, statevector
from src.base_cst import cst_max_smallest
import matplotlib.pyplot as plt


class Plotter:
    """
    Custom context manager to handle shifting and plotting more clearly
    """
    def __init__(self, circuit: QuantumCircuit, axes: Sequence[QuantumRegister], colors: Sequence[int], shift_method: Callable):
        if (dims := len(axes)) not in (2, 3): 
            raise NotImplementedError(f'Cannot plot {dims}D structures.')

        self.method = shift_method
        self.circ = circuit.copy()
        self.axes = axes
        self.a_colors = colors  # og image color/point
        self.p_colors: dict[tuple[int], int] = self.a_colors.copy()  # shifted image color/point
        self.apriori = statevector(self.circ)  # keep a record of the original image


    def __enter__(self):
        return self


    def __exit__(self, *args):
        posteriori = statevector(self.circ)  # snapshot the shifted image
        self.plot(self.apriori, self.a_colors, f'{self.method.__name__}: Before')  # plot original
        self.plot(posteriori, self.p_colors, f'{self.method.__name__}: After')     # plot shifted
        self.circ.draw('mpl')  # plot circuit
        plt.title(f'Method: {self.method.__name__} power of two')
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
        for dim, scalar in enumerate(axis): 
            self.method(self.circ, self.axes[dim], step * scalar)

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
            for sublist, pi in zip(coords, point, strict=True): sublist.append(pi)

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(*reversed(coords), c=cols)

        return fig


def experiment_2D (size: int):
    """
    Generates the two diagonals of a square of side `size`.
    """
    return encoding(2, size, 
        lambda x, y: x == y, 
        lambda x, y: x == size - 1 - y)


def experiment_3D (size: int):
    """
    Generates the four diagonals of a cube of side `size`.
    """
    return encoding(3, size, 
        lambda x, y, z: x == y == z, 
        lambda x, y, z: x == y == size - 1 - z,
        lambda x, y, z: x == size - 1 - y == z,
        lambda x, y, z: x == size - 1 - y == size - 1 - z)


if __name__ == '__main__':
    with Plotter(*experiment_2D(8), cst_max_smallest) as pltr:
        """ 2D example """
        # move three steps along the x/3=y/2 axis
        # this implies 3x3=9 steps along the x-axis and 3x2=6 steps along the y-axis
        # however, since 9 mod 8 = 1 (8 is the image side size), the x-axis step will actually be 1
        pltr.shift([3, 2], step=3)

    with Plotter(*experiment_3D(8), cst_max_smallest) as pltr:
        """ 3D example """
        # move one step along the x/3=y/2=z axis
        # this implies 1x3=3 steps along the x-axis, 1x2=2 along the y-axis and 1x1=1 along the z-axis
        pltr.shift([3, 2, 1], step=1)
        