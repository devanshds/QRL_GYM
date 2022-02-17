import cirq
import numpy as np
from qrl_gym.envs.QAS import QuantumArchSearch
from qrl_gym.utils import *

class NQubit(QuantumArchSearch):
    def __init__(self, 
                target: np.ndarray, 
                fidelity_threshold: float = 0.95, 
                reward_penalty: float = 0.01,
                max_timesteps: int = 20
    ):
        n_qubits = int(np.log2(len(target)))
        qubits = cirq.LineQubit.range(n_qubits)
        state_observables = get_default_observables(qubits)
        action_gates = get_default_gates(qubits)
        super(NQubit,self).__init__(target, qubits, state_observables, action_gates,fidelity_threshold, reward_penalty, max_timesteps)