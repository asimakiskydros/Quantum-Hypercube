from qiskit import QuantumCircuit, QuantumRegister
from src.mytools import cycle_shift, render, sgn_mag
from math import log2


def cst_max_smallest (circuit: QuantumCircuit=None, pos: QuantumRegister=None, step: int=1, show=False, barrier=False, width: int=None):
    """
    Shifts mod 2**len(pos) along the dimension expressed by `pos`, for the given amount of `step`s.

    To do so, it decomposes `step` into a sum of powers of two corresponding to the indices in the bit representation of `step`
    where `step` has bit value '1'. E.g. 30 = 16 + 8 + 4 + 2 = 2^4 + 2^3 + 2^2 + 2^1 since bin(30)=b'011110'.
    Then each power of two is realized as a cycle shift, saving on gates for the total `step`.

    The `width` parameter enables testing mode.
    """
    if (circuit is None or pos is None) and width is None:  # for testing the qiskit stuff may be omitted
        raise Exception('Empty `circuit` or `pos` arguments are allowed only during testing (`width` was found None).')
    
    record = [] if width is not None else None
    sgn, step = sgn_mag(step, pos.size if pos is not None else width)
    # can only truly shift by mod 2**n jumps, so normalize the step to this region
    # if the remaining step is 0, do nothing
    while step != 0:
        if circuit is not None and barrier: circuit.barrier()
        # jump by the largest power of 2 thats less than the remaining step
        jump = 1 << (step.bit_length() - 1)
        if record is None: cycle_shift(circuit, pos, jump, sgn > 0)  # for testing the actual circuit needs not be built
        step -= jump
        if record is not None: record.append(width - int(log2(jump)))

    if circuit is not None and show: 
        render(circuit)

    return record
    