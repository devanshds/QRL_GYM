from typing import List
import cirq
import numpy as np

def get_default_gates(
        qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    gates = []
    for idx, qubit in enumerate(qubits):
        next_qubit = qubits[(idx + 1) % len(qubits)]
        gates += [
            cirq.rz(np.pi / 4.)(qubit),
            cirq.X(qubit),
            cirq.Y(qubit),
            cirq.Z(qubit),
            cirq.H(qubit),
            #cirq.CNOT(qubit, next_qubit),
        ]
        for idx1,qubit1 in enumerate(qubits):
            for idx2,qubit2 in enumerate(qubits):
                if idx2==idx1:
                    continue
                gates += cirq.CNOT(qubit1, qubit2)
    return gates

def get_default_observables(
        qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    observables = []
    for qubit in qubits:
        observables += [
            cirq.X(qubit),
            cirq.Y(qubit),
            cirq.Z(qubit),
        ]
    return observables


def get_dicke_state(n, k) -> np.ndarray:
    target = np.zeros(2**n, dtype = complex)
    i=0
    while(i<2**n):
        set_bits = bin(i).count("1")
        if set_bits == k:
            target[i] = 1./np.sqrt(k) + 0.j
        i+=1
    return target