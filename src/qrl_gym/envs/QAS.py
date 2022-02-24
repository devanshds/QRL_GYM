import sys
from contextlib import closing
from io import StringIO
from typing import List
import cirq
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class QuantumArchSearch(gym.Env):
    metadata = {'render.modes': ['ansi', 'human']}

    def __init__(self,
        target: np.ndarray,
        qubits: List[cirq.LineQubit],
        state_observables: List[cirq.GateOperation],
        action_gates: List[cirq.GateOperation],
        fidelity_threshold: float,
        reward_penalty: float,
        max_timesteps: int,
    ):
        super(QuantumArchSearch, self).__init__()
        self.target = target
        self.qubits = qubits
        self.state_observables = state_observables
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps

        self.target_density = target * np.conj(target).T
        self.simulator = cirq.Simulator()

        self.observation_space = spaces.Box(low=-1.,
                                            high=1.,
                                            shape=(len(state_observables), ))
        self.action_space = spaces.Discrete(n=len(action_gates))
        self.circuit_gates = []
        self.seed()


    def __str__(self):
        desc = 'QuantumArchSearch-v0('
        desc += '{}={}, '.format('Qubits', len(self.qubits))
        desc += '{}={}, '.format('Target', self.target)
        desc += '{}=[{}], '.format(
            'Gates', ', '.join(gate.__str__() for gate in self.action_gates))
        desc += '{}=[{}])'.format(
            'Observables',
            ', '.join(gate.__str__() for gate in self.state_observables))
        return desc

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.circuit_gates = []
        return self.get_obs()

    def get_cirq(self):
        circuit = cirq.Circuit(cirq.I(qubit) for qubit in self.qubits)
        for gate in self.circuit_gates:
            circuit.append(gate)
        return circuit

    def get_obs(self):
        circuit = self.get_cirq()
        obs = self.simulator.simulate_expectation_values(
            circuit, observables=self.state_observables)
        return np.array(obs).real

    def get_fidelity(self):
        circuit = self.get_cirq()
        pred = self.simulator.simulate(circuit).final_state_vector
        fidelity = cirq.qis.fidelity(pred, self.target)
        return fidelity

    def step(self, action):
        #Preprocessing
        n=len(self.qubits)
        qubit_gates = {}
        for i in range(0,n):
            qubit_gates[i]=[]
            
        gates = []
        for i in self.circuit_gates:
            gates.append(str(i))

        for s in gates:
            for c in reversed(s):
                if c==')':
                    continue
                elif c.isdigit():
                    qubit_gates[int(c)].append(s)
                elif c == '(' or c == ',':
                    break

        #Check if samee action is being taken twice
        action_gate = self.action_gates[action]

        for c in reversed(str(action_gate)):
            if c==')':
                continue
            elif c.isdigit():
                pos = int(c)
            elif c == '(' or c == ',':
                break
        if len(qubit_gates[pos]) != 0:
            if qubit_gates[pos][-1] == str(action_gate) and not str(action_gate).startswith('Rz'):
                return self.get_obs(), -2*self.reward_penalty, False, {'fidelity': self.get_fidelity(), 'circuit': self.get_cirq()}
        
        #Actual step function
        self.circuit_gates.append(action_gate)

        observation = self.get_obs()

        fidelity = self.get_fidelity()

        if fidelity > self.fidelity_threshold:
            reward = fidelity - self.reward_penalty
        else:
            reward = -self.reward_penalty

        terminal = (reward > 0.) or (len(self.circuit_gates) >=
                                     self.max_timesteps)

        info = {'fidelity': fidelity, 'circuit': self.get_cirq()}

        return observation, reward, terminal, info

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n' + self.get_cirq().__str__() + '\n')

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()