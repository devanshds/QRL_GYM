from gym.envs.registration import register

register(id='QAS-v0',
         entry_point='qrl_gym.envs:QuantumArchSearch',
         nondeterministic=True)

register(id='NQubit-v0',
         entry_point='qrl_gym.envs:NQubit',
         nondeterministic=True)