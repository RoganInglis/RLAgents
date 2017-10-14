import numpy as np


class ExperienceReplayBuffer:
    # TODO comment
    def __init__(self, size, observation_shape):
        self.size = size

        # Make sure shape is list
        observation_shape = list(observation_shape)

        # Get replay shapes
        observation_shape.insert(0, size)

        # Initialise experience with zeros TODO - May be better to store action as 1 hot bool or small int array?
        self.experience = {'observation_t': np.zeros(observation_shape),
                           'action_t': np.zeros(size),
                           'observation_t_1': np.zeros(observation_shape),
                           'reward_t_1': np.zeros(size),
                           'done': np.zeros(size)}

        # Initialise fill counter and sample indices
        self.fill_counter = 0
        self.sample_indices = np.array([], np.int)

    def add(self, observation_t, action_t, observation_t_1, reward_t_1, done):
        """
        Add observation, action and subsequent reward to replay buffer
        :param observation_t: observation at time t
        :param action_t: action at time t
        :param observation_t_1: observation at time t + 1
        :param reward_t_1: reward at time t + 1
        :param done: bool indicating the termination status of the episode
        """
        # Determine index at which to add experience
        if self.fill_counter < self.size:
            # Using an array of indices so that the whole of the memory doesn't need to be copied and overwritten
            self.sample_indices = np.append(self.sample_indices, self.fill_counter)
            add_index = self.sample_indices[-1]

            self.fill_counter += 1
        else:
            # Roll sample indices
            self.sample_indices = np.roll(self.sample_indices, -1)
            add_index = self.sample_indices[-1]

        self.experience['observation_t'][add_index] = observation_t
        self.experience['action_t'][add_index] = action_t
        self.experience['observation_t_1'][add_index] = observation_t_1
        self.experience['reward_t_1'][add_index] = reward_t_1
        self.experience['done'][add_index] = done

    def get_batch(self, batch_size, fixed_batch_size=False):
        # TODO comment
        # Get batch indices
        if fixed_batch_size and self.fill_counter == 0:
            batch_indices = np.zeros(batch_size, np.int)
        elif not fixed_batch_size and self.fill_counter == 0:
            batch_indices = np.zeros(1, np.int)
        elif not fixed_batch_size and self.fill_counter < batch_size:
            batch_indices = np.random.choice(self.sample_indices, self.fill_counter)
        else:
            batch_indices = np.random.choice(self.sample_indices, batch_size)

        # Sample from replay buffer
        observation_t = self.experience['observation_t'][batch_indices]
        action_t = self.experience['action_t'][batch_indices]
        observation_t_1 = self.experience['observation_t_1'][batch_indices]
        reward_t_1 = self.experience['reward_t_1'][batch_indices]
        done = self.experience['done'][batch_indices]

        return observation_t, action_t, observation_t_1, reward_t_1, done

    def get_batch_feed_dict(self, batch_size, placeholders_dict, fixed_batch_size=False):
        # TODO comment
        # Get batch
        observation_t, action_t, observation_t_1, reward_t_1, done = self.get_batch(batch_size, fixed_batch_size)

        # Convert to feed dict
        feed_dict = {placeholders_dict['observation_t']: observation_t,
                     placeholders_dict['action_t']: action_t,
                     placeholders_dict['observation_t_1']: observation_t_1,
                     placeholders_dict['reward_t_1']: reward_t_1,
                     placeholders_dict['done']: done}

        return feed_dict


if __name__ == '__main__':
    buffer_size = int(4)
    episodes = 2
    import gym
    from pympler import asizeof

    env = gym.make('CartPole-v0')
    observation_shape = env.observation_space.shape

    replayBuffer = ExperienceReplayBuffer(buffer_size, observation_shape)

    # Fill buffer
    for episode in range(episodes):
        observation_t = env.reset()
        done = False

        while not done:

            print(replayBuffer.get_batch(16))

            action_t = env.action_space.sample()

            observation_t_1, reward_t_1, done, _ = env.step(action_t)

            replayBuffer.add(observation_t, action_t, observation_t_1, reward_t_1, done)

            observation_t = observation_t_1

    mem_size = asizeof.asizeof(replayBuffer)
    batch = replayBuffer.get_batch(16)
    print(batch)



