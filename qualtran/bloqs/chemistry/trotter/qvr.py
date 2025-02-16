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
"""Quantum Variable Rotation."""

from functools import cached_property
from typing import Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, Register, Signature
from qualtran.bloqs.basic_gates import Rz
from qualtran.bloqs.basic_gates.rotation import RotationBloq
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class QuantumVariableRotation(Bloq):
    r"""Bloq implementing Quantum Variable Rotation

    $$
        \sum_j c_j|\phi_j\rangle \rightarrow \sum_j e^{i \xi \phi_j}  c_j | \phi_j\rangle
    $$

    This is the basic implementation in Fig. 14 of the reference.

    Args:
        bitsize: The number of bits encoding the phase angle $\phi_j$.

    Register:
        phi: a bitsize size register storing the angle $\phi_j$.

    References:
        (Faster quantum chemistry simulation on fault-tolerant quantum
            computers)[https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta]
            Fig 14.
    """
    phi_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('phi', bitsize=self.phi_bitsize)])

    def short_name(self) -> str:
        return 'e^{i*phi}'

    def t_complexity(self) -> 'TComplexity':
        # Upper bounding for the moment with just phi_bitsize * Rz rotation gates.
        return self.phi_bitsize * Rz(0.0).t_complexity()

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        theta = ssa.new_symbol('theta')
        # need to update rotation bloq.
        return {(RotationBloq(theta), self.phi_bitsize)}
