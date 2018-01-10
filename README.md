# RLAgents

Implementations of reinforcement learning algorithms in TensorFlow.

The aim is to implement each algorithm such that different Q/value/policy/representation networks can be plugged in for easy experimentation.

## Currently Implemented
### Q Learning Agents
 - DQN (DQNAgent) [Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    - Implemented with options for:
        - Double Q learning [Paper](https://arxiv.org/abs/1509.06461)
        - Prioritised experience replay [Paper](https://arxiv.org/pdf/1511.05952.pdf)
        - (To be implemented next) N-step Q learning

- Current aim is to implement all of the DQN extensions used for Rainbow [Paper](https://arxiv.org/pdf/1710.02298.pdf)

        
## Requirements

Created and tested using:
- Python 3.5
- TensorFlow 1.4

#### Packages

- tensorflow
- numpy
- gym
- opencv-python
- matplotlib
- seaborn

```commandline
pip install -r requirements.txt
```

Or for GPU TensorFlow:

```commandline
pip install -r requirements-gpu.txt
```

## Usage

To train on the CartPole-v0 environment:

```commandline
python main.py
```

Additional command line arguments are detailed in main.py. This can be made to work with Atari with very minimal edits.
CartPole is the default environment currently while this is being developed but the default will be switched to Atari
once everything is implemented and tested.

## Results

Mean test episode length during training on CartPole-v0 with double Q-learning and prioritised experience replay 
enabled (with minimal hyperparameter search performed):

![Mean Episode Length](images/cartpole_ep_len.png?raw=true "Mean test episode length")

## Notes

### TODO

- [ ] Create experience replay buffer within TensorFlow
- [ ] Refactor to use an 'observe' function, which should be the agents only interaction outside of TensorFlow
- [ ] Complete implementation of Rainbow
    - [ ] Implement n-step Q-learning
    - [ ] Implement distributional RL
    - [ ] Implement duelling networks
    - [ ] Implement noisy nets
    - [ ] Test on Atari
- [ ] Implement policy gradient agents (A2C, DDPG, PPO)
- [ ] Implement [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
- [ ] Implement [Curiosity Driven Exploration by Self-Supervised Prediction](https://arxiv.org/abs/1705.05363)
    
   


### Reference

1. [Human Level Control Through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
2. [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
3. [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
4. [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf)
5. [Project structure](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3)
