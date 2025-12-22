from qiskit import QuantumCircuit, QuantumRegister
from src.mytools import cycle_shift, render, sgn_mag
from math import log2


def cst_nearest (circuit: QuantumCircuit=None, pos: QuantumRegister=None, step: int=1, show=False, barrier=False, width: int=None):
    """
    []========================================== A.K.A. ALGORITHM 2 ==========================================[]

    Shifts mod 2**len(pos) along the dimension expressed by `pos`, for the given amount of `step`s.

    To do so, it attempts to cover as much distance as possible by realizing a cycle shift covering distance equal to the nearest 
    power of two from the current `step` value. This idea stems from two remarks:
    1. Shifting forwards or backwards by a specific power of two has no effect on the cost, since the sign is enforced by simply
    reflecting the gate setup. The amount of gates stays the same.
    2. Larger powers of two and thus larger jumps result in less gates.

    Therefore, you can greed by selecting the nearest power of two from the current step remainder, favoring the larger one on a tie,
    betting that the overshoot handling cost will be smaller than that of the extra distance `max_smallest` would have to cover.
    This does not always work, e.g. step = 53.
    """
    if (circuit is None or pos is None) and width is None:  # for testing the qiskit stuff may be omitted
        raise Exception('Empty `circuit` or `pos` arguments are allowed only during testing (`width` was found None).')
    
    record = [] if width is not None else None
    sgn, step = sgn_mag(step, pos.size if pos is not None else width)
    # can only truly shift by mod 2**n jumps, so normalize the step to this region
    # if the remaining step is 0, do nothing    
    while step != 0:
        if circuit is not None and barrier: circuit.barrier()
        # jump by the nearest power of 2
        lo = 1 << (step.bit_length() - 1)
        hi = 1 << step.bit_length()
        # if equal distance to either, choose the larger one, saves more gates
        jump = lo if step - lo < hi - step else hi
        if record is None: cycle_shift(circuit, pos, jump, sgn > 0)  # for testing the actual circuit needs not be built
        sgn = -sgn if jump > step else sgn
        step = abs(step - jump)
        if record is not None: record.append(width - int(log2(jump)))

    if circuit is not None and show: 
        render(circuit)

    return record
