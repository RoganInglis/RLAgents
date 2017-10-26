import numpy as np
import cv2
import gym


class ExperienceReplayBuffer:
    # TODO comment
    def __init__(self, size, observation_shape, observation_dtype=np.uint8):
        self.size = size

        # Make sure shape is list
        self.observation_shape = list(observation_shape)

        # Get replay shapes
        self.observation_shape.insert(0, size)

        # Initialise experience with zeros TODO - May be better to store action as 1 hot bool or small int array?
        self.experience = {'observation_t': np.zeros(self.observation_shape, dtype=observation_dtype),
                           'action_t': np.zeros(size),
                           'observation_t_1': np.zeros(self.observation_shape, dtype=observation_dtype),
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

    def fill_with_random_experience(self, n_to_fill, env):
        # Deal with the fact that observations from sequential timesteps may need to be stacked
        observation_t = env.reset()

        # Initialise number filled counter
        n = 0
        while n < n_to_fill:
            observation_t = env.reset()
            done = False

            # Run environment for one episode
            while n < n_to_fill and not done:
                action_t = env.action_space.sample()
                observation_t_1, reward_t_1, done, _ = env.step(action_t)

                self.add(observation_t, action_t, observation_t_1, reward_t_1, done)

                observation_t = observation_t_1
                n += 1


class FrameBuffer:
    def __init__(self, observation_shape, observations_to_stack):
        self._args = (observation_shape, observations_to_stack)
        self.shape = list(observation_shape)
        self.shape.append(observations_to_stack)

        # Initialise numpy array to hold stacked frames
        self.frames = np.zeros(self.shape)

        self.retrieve_indices = np.arange(observations_to_stack)

    def add(self, observation):
        # Roll indices
        self.retrieve_indices = np.roll(self.retrieve_indices, -1)

        # Add frame in correct location (using indices to avoid having to copy and rewrite the entire array by rolling)
        self.frames[:, :, self.retrieve_indices[-1]] = observation

    def get(self):
        return self.frames[:, :, self.retrieve_indices]

    def add_get(self, observation):
        # Add observation to buffer and then return current buffer
        self.add(observation)
        return self.get()

    def reset(self):
        self.__init__(*self._args)


class EnvWrapper(gym.Wrapper):
    """
    Wrapper to allow environment to directly output a preprocessed set of stacked consecutive frames
    Will allow the agent code to stay as general as possible
    """
    def __init__(self, env, preprocessor_func=None, frames_to_stack=1, repeat_count=1, clip_rewards=False):
        super(EnvWrapper, self).__init__(env)
        self.preprocessor_func = preprocessor_func
        self.frames_to_stack = frames_to_stack
        self.use_frame_buffer = self.frames_to_stack > 1
        self.repeat_count = repeat_count
        self.clip_rewards = clip_rewards

        # Get processed observation shape and update observation_space.shape
        if self.preprocessor_func is None:
            observation_shape = self.observation_space.shape
        else:
            test_observation = self.env.reset()
            test_observation = self.preprocessor_func(test_observation.astype(np.uint8))
            observation_shape = test_observation.shape
            if frames_to_stack > 1:
                observation_space_shape = (*observation_shape, frames_to_stack)
            else:
                observation_space_shape = observation_shape

        if type(self.observation_space) == gym.spaces.Box:
            self.observation_space = gym.spaces.Box(0, 255, observation_space_shape)
        else:
            raise Exception('The selected environment\'s observation space is not of type Box. Other types are not yet supported by the environment wrapper currently in use' )


        self.frame_buffer = FrameBuffer(observation_shape, frames_to_stack)

    def _step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < (self.repeat_count + 1) and not done:
            observation, reward, done, info = self.env.step(action)
            if self.preprocessor_func is not None:
                observation = self.preprocessor_func(observation.astype(np.uint8))

            if self.use_frame_buffer:
                self.frame_buffer.add(observation)

            if self.clip_rewards:
                if reward > 0:
                    reward = 1
                elif reward < 0:
                    reward = -1

            total_reward += reward
            current_step += 1

            observation = self.frame_buffer.get()

        return observation.astype(np.uint8), total_reward, done, info

    def _reset(self):
        observation = self.env.reset()

        if self.preprocessor_func is not None:
            observation = self.preprocessor_func(observation)

        if self.use_frame_buffer:
            self.frame_buffer.reset()
            observation = self.frame_buffer.add_get(observation)

        return observation.astype(np.uint8)


def preprocess_atari(atari_observation):
    # Convert to greyscale
    greyscale_observation = cv2.cvtColor(atari_observation, cv2.COLOR_BGR2GRAY)

    # Crop?
    cropped_observation = greyscale_observation[0:-1, 0:-1]

    # Resize
    resized_observation = cv2.resize(cropped_observation, (84, 110))

    # Convert to correct dims array
    preprocessed_atari_observation = resized_observation
    return preprocessed_atari_observation


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



