from dataclasses import dataclass
from typing import Callable, Any, Sequence, Literal, Iterable
from itertools import count
from typing_extensions import override
from qiskit import QuantumCircuit, QuantumRegister
from src.mytools import birange, is_power_of_two, sign_magnitude

# custom types
SubShift = tuple[Literal[-1, 0, 1], int]


@dataclass(frozen=True, slots=True)
class CST:
    """
    Cycle shift manager class. Subclasses define the decomposition method.
    """
    barriers: bool=False        # adds barriers in-between triangles

# []========================================================================================================================[]
#                           vvv                   CONVENIENCE FUNCTIONS                   vvv
# []========================================================================================================================[]

    @classmethod
    def methods (cls, *, barriers=False, shuffles: int=0, k: int=-1):
        """
        Returns a list of the first `k` subclasses of this class, instantiated.

        If `k` is left undeclared or assigned negative, it yields all of them.
        """
        subclasses = cls.__subclasses__()
        n = k if k > -1 else len(subclasses)  # future-proof check: should more subclasses be added, yield only the k first

        return tuple(method(barriers, shuffles) for i, method in enumerate(subclasses) if i < n)


    @property
    def name (self):
        """
        Returns the name of the object class as a `str`.
        """
        return self.__class__.__name__


    def decompose (self, step: int, width: int, f: Any=None) -> Iterable[SubShift]:
        """
        Decomposes `step` into a sequence of powers of two, to be implemented as cycle shift triangles. 

        This specific placeholder performs the most naive strategy imaginable: it moves by +/-1, `step` times.
        Actual strategies are entailed in the following subclasses.
        """
        sgn, s = sign_magnitude(step, width)
        for _ in range(s): yield sgn, 0


    def as_sequence (self, step: int, width: int, f: Any=lambda x:x) -> Iterable[SubShift]:
        """
        Maps the specified `step` into a signs and powers sequence, according to the triangle decomposition method.

        Currently just parrots `self.decompose`, placeholder for future extensions.
        """
        yield from self.decompose(step, width, f)
    

    def cost (self, steps: int | Sequence[int], width: int, cost_model: Callable=None, decomposition_model: Callable=None) -> int | list[int]:
        """
        Computes the cost of implementing the given step(s) imposed by this method for the given `width` according to the cost model `f`.

        Normally, `decomposition_model` == `cost_model`. Both can be ignored for decision-making, and this format allows for testing this point.
        """
        cost_model = cost_model or (lambda x: x)
        decomposition_model = decomposition_model or (lambda x: x)

        if isinstance(steps, int):
            # single step case                                        vvv-----(only the raw decomposition matters here)
            return sum(cost_model(width - power) for _, power in self.decompose(steps, width, decomposition_model))
        
        return [self.cost(step, width, cost_model, decomposition_model) for step in steps]


    def depth (self, steps: int | Iterable, width: int) -> int | list[int]:
        """
        Analytically calculates the total depth for the given step(s).

        * If `steps` is `int`, it treats it as the signed CST step to consider.
        * If `steps` is an `Iterable`, it treats it as the decomposed sequence of `SubShift`s for a given CST step to consider.
        """
        if isinstance(steps, int):
            return self.depth(self.as_sequence(steps, width), width)
        
        # keeps track the depth evolution of the entire host register after the application of each MCX gate (pre-transpilation)
        depths = [0] * width
        for sgn, power in steps:
            for i in birange(power, width, reverse=sgn > 0):
                d = max(depths[power: i + 1]) + 1
                for j in range(power, i + 1): depths[j] = d

        return max(depths)           
        
# []========================================================================================================================[]
#                           vvv                   STUDIED ALGORITHMS                   vvv
# []========================================================================================================================[]

    def shift (self, *, by: int=1, on: QuantumRegister | Sequence[QuantumRegister], circuit: QuantumCircuit, shift_axis: Sequence=None):
        """
        Builds the desired shift step `by` as a sequence of "power triangles", quantum operators each implementing
        a power of two cycle shift sub step. The total step follows the direction specified by the sign of `by`.
        """
        if shift_axis is not None:
            # Multidimensional shift, described in Theorem 3 of 10.1088/1402-4896/ae1d2b.
            # Performs a cycle shift of the desired step units, across the hypercube described by the given host registers, 
            # along the specified `shift_axis`. All host registers are assumed to be owned by the same, given `circuit`. 
            # To do so, it decomposes the united multidimensional shift into per-dimension sub-shifts. Each such shift is 
            # performed on its host dimension, by units of the total step scaled by the axis constant of the corresponding dimension.
            # If `len(on) != len(shift_axis)` this throws. 
            assert isinstance(on, Sequence)
            for reg, scalar in zip(on, shift_axis):
                self.shift(by=by * scalar, on=reg, circuit=circuit)
            
            return circuit

        s, m = sign_magnitude(by, on.size)
        if m != 0:  # if the power is larger the modulus of the host drives it to zero
            if self.barriers: circuit.barrier()
            for sgn, sub_step in self.as_sequence(s * m, on.size):            
                # to jump by `2^power`, start the shift triangle `power` qubits later
                # reflect the triangle vertically if the step is negative
                for i in birange(sub_step, on.size, reverse=sgn > 0):
                    if i == sub_step: circuit.x(on[i])
                    else: circuit.mcx(on[sub_step:i], on[i])
        
        return circuit
                    

class MaxSmallest(CST):
    """
    A.K.A. Alg. 1 in  https://doi.org/10.1002/qute.70445.

    Decomposes the step into the indices of those bits in its binary representation that are active.

    E.g. decimal(+30) = +binary(11110) => decomposition=[+4, +3, +2, +1] since 2^4+2^3+2^2+2^1=30.
    """
    @override
    def decompose (self, step: int, width: int, f: Any=None) -> Iterable[SubShift]:
        sgn, s = sign_magnitude(step, width)
        while s:
            power = s.bit_length() - 1  # select exponent of largest power of two at most equal to remainder `s`
            s -= (1 << power)
            if power < width: yield sgn, power


class Nearest(CST):
    """
    A.K.A. Alg. 2 in https://doi.org/10.1002/qute.70445.

    Decomposes the step according to the following strategy: at every remainder, attempt to cover as much distance as possible
    without overshooting too much. Examine both closest powers of two from the current position, lo and hi, and select that 
    which minimizes the absolute distance. The intuition behind this idea stems from the following points:
    1. Larger powers result in smaller triangle operators and thus less gates.
    2. The sign plays no role in gate count since it is effected by gate order.

    Therefore, the greedy strategy bets that the larger sub steps will lead to overshoots that cost less to handle than the
    extra distance `MaxSmallest` would have to cover. This does not always work, e.g. step=53.
    """
    @override
    def decompose (self, step: int, width: int, f: Any=None) -> Iterable[SubShift]:
        sgn, s = sign_magnitude(step, width)
        while s:
            bitlen = s.bit_length()
            lo, hi = bitlen - 1, bitlen
            # select the exponent of the nearest power of two to the current remainder by absolute distance
            # select hi on ties, saves gates
            power = lo if (s - (1 << lo)) < ((1 << hi) - s) else hi
            if power < width: yield sgn, power
            # if the condition is met, hi was selected, so the sign flips for the next remainder
            sgn = -sgn if power == bitlen else sgn
            s = abs(s - (1 << power))


class DP_DFS(CST):
    """
    A.K.A Alg. 4 (using Alg. 3) in  https://doi.org/10.1002/qute.70445.
    
    Decomposes the step by examining the entire binary weighted tree formed from the two closest powers of two from each possible remainder.
    Each node holds a remainder, the left edge is weighted by its next smallest power (lo), and the right by its next largest (hi). The nodes
    also hold weights, those being the weight of the lightest child node + the weight of the edge leading to that child. Ties favor hi. The optimal
    decomposition is the edge path that minimizes the cost of the root. The cost function `f` must be decreasing with the power, or equally, increasing 
    with the span. The tree does not consider signs, since they do not affect the cost and they are inferrable later on.
    """
    @override
    def decompose (self, step: int, width: int, f: Callable=lambda x:x) -> Iterable[SubShift]:
        sgn, s = sign_magnitude(step, width)
        stack = [s]
        m = { 0: (-1, 0, 0) }
        # perform DFS to find the optimal decomposition by `f`
        while stack:
            if stack[-1] not in m:  # memoize examined values
                bitlen = stack[-1].bit_length()
                lo, hi = bitlen - 1, bitlen
                if is_power_of_two(stack[-1]):  # if root is a power of two there are no children
                    m[stack.pop()] = (lo, 0, f(width - lo))
                    continue
                left, right = stack[-1] - (1 << lo), (1 << hi) - stack[-1]
                # prioritize any unknown children (starting from the smallest one)
                if left not in m: stack.append(left); continue
                if right not in m: stack.append(right); continue
                # children now known; return to head of stack computation
                wl = m[left][2] + f(width - lo)                                # cost of left child + SUM_{i=1}^{n-(p-1)}MCX(i)
                wr = m[right][2] + f(width - hi)                               # cost of right child + SUM_{i=1}^{n-p}MCX(i)
                m[stack[-1]] = (lo, left, wl) if wl < wr else (hi, right, wr)  # prefer lower cost. On ties, prefer larger powers (smaller circuits)
            # head of stack now known; skip
            stack.pop()
        # apply the decided sequence
        p = m.get(s)
        while p[0] != -1:
            if p[0] < width: yield sgn, p[0]
            jump = 1 << p[0]
            # the decomposition selected the larger closest power from the current remainder. Flip the sign for the next.
            sgn = -sgn if jump > s else sgn
            s = abs(s - jump)
            p = m.get(p[1])


class Domino(CST):
    """
    Decomposes the step according to its NAF that minimizes the total shift cost. To reach the NAF, it iterates over
    the bits in ascending power order and checks for adjacent active bits. Each such pair implies +-(2^{k+1}+2^k),
    whereas the equivalent +-(2^{k+2}-2^k) is cheaper. This is called an R4 invocation. Eliminating all such cases 
    in the order specified yields the optimal sequence. The spawned higher power, k+2, is propagated to later bits,
    potentially causing a "domino" effect to any existing adjacent ace chains.

    E.g. 0b10110111 -> 0b101101[11] -> 0b1011100(-1) -> 0b101[11]00(-1) -> 0b1100(-1)00(-1) -> 0b[11]00(-1)00(-1) -> 0b10(-1)00(-1)00(-1)
    so the sequence becomes [+8, -6, -3, -0].

    Equivalent to `DP_DFS` in output but better in performance.
    """
    @override
    def decompose (self, step: int, width: int, f: Any=None) -> Iterable[SubShift]:
        sgn, s = sign_magnitude(step, width)
        for bitlen in count():
            if not s: break
            direction = sgn
            # pop the least significant bit from the current remainder s
            lsb = s & 1
            s >>= 1
            if lsb == 1:
                # if both the old lsb and the new lsb are active, there is an R4 invocation
                if s & 1 == 1:
                    direction = -direction  # flip the sign of the current sub step 
                    s += 1                  # the current power has been popped. Adding 1 "dominos" the R4 elimination carry to later bits
                if bitlen < width: yield direction, bitlen
