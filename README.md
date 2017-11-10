# RLAgents

Implementations of reinforcement learning algorithms using TensorFlow.

The aim is to implement each algorithm such that different Q/value/policy/representation networks can be plugged in for easy experimentation.

## Currently Implemented
### Q Learning Agents
 - DQN (DQNAgent) [Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    - Implemented with options for:
        - Double Q learning [Paper](https://arxiv.org/abs/1509.06461)
        - Prioritised experience replay [Paper](https://arxiv.org/pdf/1511.05952.pdf)
        - (To be implemented next) N-step Q learning

- Current aim is to implement all of the DQN extensions used for Rainbow [Paper](https://arxiv.org/pdf/1710.02298.pdf)

        
## How To Run
To run the agents, run main.py using optional arguments, detailed in the script, if required.