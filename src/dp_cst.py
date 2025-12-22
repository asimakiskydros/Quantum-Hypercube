from qiskit import QuantumCircuit, QuantumRegister
from src.mytools import cycle_shift, render, sgn_mag, is_power_of_two
from typing import Callable
from math import log2


def cst_dfs (circuit: QuantumCircuit=None, pos: QuantumRegister=None, step: int=1, f=lambda x: x * (x + 1) // 2, show=False, barrier=False, width: int=None):
    """
    []========================================== A.K.A. ALGORITHM 4 ==========================================[]

    Shifts mod 2**len(pos) along the dimension expressed by `pos`, for the given amount of `step`s.

    To do so, it decomposes the overall `step` into a binary weighted tree where each right child is the next biggest
    power of two from the current node (the current remaining step) and the left child the next smallest. These powers are encoded
    as edge weights. Each node is also weighted, by the minimum weight of its children + the cost to realize the lightest child's 
    corresponding edge weight. In case of weight tie, it favors the larger power (smaller circuit, saves gates). The cost function 
    should be parameterized, here assumed MCX(j) = O(j) = j for a mcx spanning j indices. Decomposition ends on cycle shifts by 1, 
    which carry the maximum possible weight. Following the edge path of smallest weight yields the optimal cycle shift amounts. 
    The tree does not consider signs, since they do not matter on the cost calculation, for further compactness, and they can be easily
    inferred when needed.

    The `width` parameter enables testing mode.
    """
    if (circuit is None or pos is None) and width is None:  # for testing the qiskit stuff may be omitted
        raise Exception('Empty `circuit` or `pos` arguments are allowed only during testing (`width` was found None).')
    
    record = [] if width is not None else None
    w = pos.size if pos is not None else width
    sgn, step = sgn_mag(step, w)
    m = decompose(step, w, f)  # decomposes with the largest even factor extracted from `step`
    path = m.get(step)
    # traverse the resulting chain starting from path := m[step], path[0] holds the power of two jump and path[1] the next chain link
    while path[0] != 0:
        if circuit is not None and barrier: circuit.barrier()
        # always reduce by absolute value: if remaining step is positive, go forwards (subtract from the total), otherwise go backwards
        if record is None: cycle_shift(circuit, pos, path[0], sgn > 0)  # for testing the actual circuit needs not be built
        sgn = -sgn if path[0] > step else sgn  # flip sign if the power used is the larger nearest one
        step = abs(step - path[0])
        if record is not None: record.append(width - int(log2(path[0])))
        path = m.get(path[1])

    if circuit is not None and show: 
        render(circuit)

    return record


def decompose (distance: int, width: int, f: Callable) -> tuple[int, int, int]:
    """
    []========================================== A.K.A. ALGORITHM 3 ==========================================[]

    Decomposes `distance` into a sum of powers with minimal total `Σf`. At every remaining step, it studies both nearest 
    powers of two, to see which choice leads to a path with the smallest overall gate cost. Sign is not saved because
    it is effected by circuit reflection, therefore it does not contribute to the overall cost. Assumes `distance` is
    scaled down.
    """
    stack = [distance]
    m = { 0: (0, 0, 0) }
    while stack:
        if stack[-1] not in m:  # memoize examined values
            if is_power_of_two(stack[-1]):  # if root is a power of two there are no children
                m[stack[-1]] = (stack[-1], 0, f(width - stack[-1].bit_length() + 1))
                stack.pop()
                continue
            lo, hi = 1 << (stack[-1].bit_length() - 1), 1 << stack[-1].bit_length()
            left, right = stack[-1] - lo, hi - stack[-1]
            # prioritize any unknown children (starting from the smallest one)
            if left not in m:
                stack.append(left)
                continue
            if right not in m:
                stack.append(right)
                continue
            # children now known; return to head of stack computation
            wl = m[left][2] + f(width - stack[-1].bit_length() + 1)        # cost of left child + SUM_{i=1}^{n-(p-1)}MCX(i)
            wr = m[right][2] + f(width - stack[-1].bit_length())           # cost of right child + SUM_{i=1}^{n-p}MCX(i)
            m[stack[-1]] = (lo, left, wl) if wl < wr else (hi, right, wr)  # prefer lower cost. On ties, prefer larger powers (smaller circuits)
        # head of stack now known; skip
        stack.pop()
    return m
