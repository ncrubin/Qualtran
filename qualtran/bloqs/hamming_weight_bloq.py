#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import numpy as np
from collections import deque, defaultdict

from attrs import field, frozen
import itertools
from typing import Any, Dict, Set, Tuple

from qualtran import  Bloq, BloqBuilder, Register, Side, Signature
from qualtran.bloqs.arithmetic import OutOfPlaceAdderBuildingBlock
from qualtran.bloqs.basic_gates import CNOT

@frozen
class HammingWeight(Bloq):
    r"""Compute the Hamming weight into ancilla registers with OutOfPlaceAdders

    This bloq implements $U|k\rangle = |w(k)\rangle$ where $w$ is the Hamming 
    weight of the argument in binary.

    Note: 
    For input register size that is even, the system register passes through without modification.
    For input register size that is odd, the last bit of the system register stores the hamming weight

    Args:
        bitsize: input register bitsize

    Registers:
        n: input register we will compute Hamming weight of.
        weight [right]: The output Hamming weight register that is always $\log_{2}(n)$ in size
        junnk [right]: result of intermediate computation size $n - \log_{2}(n)$

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
        [Improved Fault-Tolerant Quantum Simulation of Condensed-Phase Correlated Electrons via Trotterization](https://quantum-journal.org/papers/q-2020-07-16-296/)
    """
    bitsize: int

    @property
    def signature(self):
        return Signature(
            [
                Register(name='n', bitsize=self.bitsize, side=Side.THRU),
                Register(name='weight', bitsize=n.bit_length(), side=Side.RIGHT),
                Register(name='ancilla', bitsize=n - n.bit_count(), side=Side.RIGHT)
            ]
        )

    def short_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"HammingWeight{dag}"

    def build_composite_bloq(self, bb: BloqBuilder, *, n):
        cwb = bb.split(n) # current_weight_bits
        srb = [] # system_register_bits
        hwb = bb.allocate(self.bitsize.bit_length()) # hamming_weight_bits
        ancilla_bits = [] 

        for w_idx in range(n.bit_length()):
            next_weight_bits = []
            for cidx in range(0, len(cwb) - 2, 2):
                cwb[cidx], cwb[cidx + 1], cwb[cidx + 2], anc = bb.add(OutOfPlaceAdderBuildingBlock(), 
                                      a=cwb[cidx], 
                                      b=cwb[cidx + 1], 
                                      c=cwb[cidx + 2])
                ancilla_bits.append(anc)
                next_weight_bits.append(anc)
                if w_idx == 0:
                    srb += [cwb[cidx], cwb[cidx + 1]]
            # if odd-n then last bit bit current_weight_bits[cidx + 2] is Hamming weight-0-bit
            # last system bit needs to be assigned to hamming weight bit
            if len(cwb) % 2 == 1: # odd case
                # do cnot into output bit indexed by w_idx to store the ouput
                hwb[w_idx], cwb[cidx + 2] = bb.add(CNOT(), ctrl=cwb[cidx + 2], target=hwb[w_idx])

                # and store the system bit to joint later
                if w_idx == 0:
                    srb.append(cwb[cidx + 2])
            # for the case where we have an even number of current bits we 
            # will have 2 remaining bits. The hamming weight is stored in the ancilla
            else: # even case - need ancilla
                a, b = cwb[-2:]
                a, b, hwb[w_idx], abit = bb.add(OutOfPlaceAdderBuildingBlock(), a=a, b=b, c=hwb[w_idx])
                next_weight_bits.append(abit)

                # if even-n then all bits have been mapped to output
                if w_idx == 0:
                    srb += [a, b]

            # All code above handles len(current_weight_bits) >= 2
            # if we have 1 bit then this stores the binary representation 
            # of the Hamming weight of 1 bit. 
            if len(cwb) == 1:
                hwb[w_idx] = cwb[0]

            # update the weight to the next set of weight qubits just generated
            cwb = next_weight_bits

        return {'n': bb.join(srb), 'weights': bb.join(hwb), 'ancilla': bb.join(ancilla_bits)}




if __name__ == "__main__":

    n = 3
    current_weight_bits = [str(x) for x in range(n)]
    print(f"{current_weight_bits=}")
    print("alpha - w(m) = ", n - np.binary_repr(n).count('1'))
    print("alpha - w(m) = ", n - n.bit_count())
    total_additions = 0
    total_ancilla = 0
    ancilla_bit_counter = n - n.bit_count()

    ancilla_bits = []
    system_register_bits = []
    hamming_weight_bits = []

    for w_idx in range(n.bit_length()):
        print()
        print(f"{w_idx}")
        next_weight_bits = []
        for cidx in range(0, len(current_weight_bits) - 2, 2):
            a, b, c = current_weight_bits[cidx], current_weight_bits[cidx + 1], current_weight_bits[cidx + 2] 
            print(a, b ,c)
            # adder goes here
            total_additions += 1
            total_ancilla += 1
            ancilla_bit = f"({a}+{b}+{c})_{2**(w_idx + 1)}" # from adder
            ancilla_bit_counter -= 1
            ancilla_bits.append(ancilla_bit)

            next_weight_bits.append(ancilla_bit)
            if w_idx == 0:
                system_register_bits += [a, b]
        else:
            # if odd-n then last bit for w_idx is Hamming weight-0-bit
            # last system bit needs to be assigned later
            if len(current_weight_bits) % 2 == 1:
                # allocate an output bit and do it
                # do cnot into output bit indexed by w_idx
                # and store the alst system bit
                if w_idx == 0:
                    system_register_bits.append("S+" + c)

        # for the case where we have an even number of current bits we 
        # will have 
        if len(current_weight_bits) % 2 == 0:
            c = f"A{2**w_idx}"
            a, b = current_weight_bits[-2:]
            print(a, b, c)
            # adder goes here
            total_additions += 1
            total_ancilla += 1
            ancilla_bit = f"({a}+{b}+{c})_{2**(w_idx + 1)}" # from adder
            ancilla_bit_counter -= 1
            ancilla_bits.append(ancilla_bit)

            next_weight_bits.append(ancilla_bit)

            # if even-n then all bits have been mapped to output
            if w_idx == 0:
                system_register_bits += [a, b]

        # All code above handles len(current_weight_bits) >= 2
        # if we have 1 bit then this stores the binary representation 
        # of the Hamming weight of 1 bit. 
        if len(current_weight_bits) == 1:
            c = current_weight_bits[0]

        # store hamming weight bit 
        hamming_weight_bits.append(c)

        # update the weight to the next set of weight qubits just generated
        current_weight_bits = next_weight_bits

    print()
    print("alpha - w(m) = ", n - np.binary_repr(n).count('1'))
    print(f"{total_additions=}")
    print(f"{total_ancilla=}")
    print(f"{ancilla_bit_counter}")
    print(system_register_bits, len(system_register_bits))
    print(hamming_weight_bits, len(hamming_weight_bits))
    print(ancilla_bits, len(ancilla_bits))
    print(f"{n=}")
    print(f"{n.bit_length()=}")
    print(f"{n - n.bit_count()=}")
    

    exit(0) 

    # exit()
    # # construct weight 1 bit
    # # (3, 1), (5, 3), (7, 3), (9, 7), (11, 8), (13, 10), (15, 11), (17, 15)
    # # (2, 1), (4, 3), (6, 4), (8, 7), (10, 8), (12, 10), (14, 11), (16, 15)
    # # for n in range(2, 32):

    # n = 16
    # qubit_weights = defaultdict(list)
    # qubit_weights[1] = [str(x) for x in range(n)]
    # number_of_adds = 0
    # ancilla_bits = []
    # summed_bits = []
    
    # print(int(np.floor(np.log2(n))), n.bit_length())

    # # for each weight 
    # for w_idx in range(int(np.floor(np.log2(n)))):
    #     # print(f"Summing weight = {2**w_idx}")
    #     # put them in the queue
    #     q = deque()
    #     q.extend(qubit_weights[2**w_idx])
    #     qubit_weights[2**w_idx] = []

    #     while len(q) > 1:
    #         if len(q) > 2:
    #             a, b, c = q.popleft(), q.popleft(), q.popleft()
    #             if w_idx == 0:
    #                 summed_bits.append(f"{a}")
    #                 summed_bits.append(f"{b}")
    #         elif len(q) == 2:
    #             q.append(f"A_{2**w_idx}")
    #             a, b, c = q.popleft(), q.popleft(), q.popleft()
    #         number_of_adds += 1
    #         qubit_weights[2**(w_idx + 1)].append(f"({a}+{b}+{c})_{2**(w_idx+1)} ")
    #         q.appendleft(f"({a}+{b}+{c})_{2**w_idx} ")
    #         # summed_bits.append(f"{a}")
    #         # summed_bits.append(f"{b}")
    #         ancilla_bits.append(f"({a}+{b}+{c})_{2**(w_idx+1)} ")
    #         # print(len(q), q, qubit_weights[2**(w_idx+1)])
    #         # print()
    #     print(w_idx, number_of_adds)

    #     qubit_weights[2**w_idx] = q.pop()
    #     print(qubit_weights[2**w_idx])
    #     print(summed_bits)
    #     exit()

    # print('------------------------------')
    # for w, summs in qubit_weights.items():
    #     print(w)
    #     print(summs)
    #     print()
    # print(f"{ancilla_bits=}")

    # print(f"{number_of_adds=}", "alpha - w(m) = ", n - np.binary_repr(n).count('1'), f"{len(ancilla_bits)=}", f"{len(qubit_weights)=}", len(np.binary_repr(n)))
    # print(f"{len(summed_bits)=}")
    # print(summed_bits)
    # assert number_of_adds == n - np.binary_repr(n).count('1')
    # # assert number_of_adds == len(ancilla_bits)
