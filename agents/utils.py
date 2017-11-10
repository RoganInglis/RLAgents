import numpy as np
import cv2
import gym
import matplotlib.pyplot as plt
import seaborn as sns


class ExperienceReplayBufferM:
    """
    A replay buffer to store and present experience for reinforcement learning algorithms.
    This experience replay buffer is optimised for minimal memory use by only storing each observation once, rather than
    once each as observation t and observation t + 1. Because of this, and because dynamically altering the size of
    numpy arrays containing the data is avoided, it will not contain exactly 'size' number of transitions but a fraction
    of this depending on the mean episode length, i.e. it will contain close to size * (1 - 1/mean episode length)
    transitions.

    Currently (although there may be further optimisations that can be done) this replay buffer is around 1.6x slower
    for adding experience compared to the non-memory optimised version (1.39e-4s vs 0.44e-4s) although appears to be
    marginally faster (~0.93x or 1.90e-4s vs 2.05e-4s) when retrieving a batch. However it uses 0.5x the amount of
    memory (for DQN atari data). This memory usage may be reduced further by updating the _update_experience_indices and
    the get_batch functions such that the atari observations may be stored as individual frames (i.e. [84, 84, 1]) but
    retrieved as stacked frames ([84, 84, 4]) rather than storing stacked frames for each observation
    """
    def __init__(self, size, observation_shape, observation_dtype=np.uint8, action_dtype=np.uint8,
                 reward_dtype=np.int16, done_dtype=np.uint8):
        self.size = size

        # Make sure shape is list
        self.observation_shape = list(observation_shape)

        # Get replay shapes
        self.observation_shape.insert(0, size)

        # Initialise experience with zeros
        self.experience = {'action_t': np.zeros(size, dtype=action_dtype),
                           'observation_t_1': np.zeros(self.observation_shape, dtype=observation_dtype),
                           'reward_t_1': np.zeros(size, dtype=reward_dtype),
                           'done': np.zeros(size, dtype=done_dtype)}

        # Using separate arrays of indices so that we don't have to store observations twice as o_t and o_t_1
        # this is slightly wasteful of memory as all but observation_t should be the same, but it makes the code easier
        # to understand and the additional memory usage should be minimal compared to the memory used for the actual
        # experience
        self.experience_indices = {'t': np.array([], np.int),
                                   't_1': np.array([], np.int)}

        # Initialise fill counter and sample indices
        # fill counter will keep track of how many observations have been stored
        self.fill_counter = 0
        self.prev_done = False
        self.next_done = False
        self.next_next_done = False
        self.filling = True
        self.batch_indices = np.zeros(1, np.int)

        # TODO - for debugging \/\/
        #self.fig, self.ax = plt.subplots(1, 4)

    def add(self, observation_t, action_t, observation_t_1, reward_t_1, done):
        """
        Add observation, action and subsequent reward to replay buffer
        :param observation_t: observation at time t - only used for the first transition of an episode
        :param action_t: action at time t
        :param observation_t_1: observation at time t + 1
        :param reward_t_1: reward at time t + 1
        :param done: bool indicating the termination status of the episode
        """
        # Determine indices at which to add experience
        # Update indices
        self._update_experience_indices()

        if self.prev_done or (self.fill_counter == 0):
            # For start of episode we need to add both the initial observation as well as the following
            # action, observation, reward and done

            # Add observation to correct place in observation array
            self._add_to_experience(self.experience_indices['t'][-1], 0, observation_t, 0, False)

            if self.filling:
                self.fill_counter += 1  # Need to add another 1 to fill counter here as we are adding 2 observations
            self.prev_done = False

        # Add action, observation, reward and done to correct place in observation array
        self._add_to_experience(self.experience_indices['t_1'][-1],
                                action_t, observation_t_1, reward_t_1, done)

        if done:
            self.prev_done = True
        if self.filling:
            self.fill_counter += 1
            if self.fill_counter >= self.size:
                self.filling = False

        # Display data for debugging purposes
        # TODO - for debugging
        #plot_replay_buffer_data(self.experience, self.experience_indices, self.fig, self.ax)

    def get_batch(self, batch_size, fixed_batch_size=False):
        # TODO comment
        # Updating batch indices as a property so they can be used to update the sampling priorities if required
        self._get_batch_indices(batch_size, fixed_batch_size)

        # Sample from replay buffer - use batch indices to get the correct indices from the experience_indices arrays
        observation_t = self.experience['observation_t_1'][self.experience_indices['t'][self.batch_indices]]
        action_t = self.experience['action_t'][self.experience_indices['t_1'][self.batch_indices]]
        observation_t_1 = self.experience['observation_t_1'][self.experience_indices['t_1'][self.batch_indices]]
        reward_t_1 = self.experience['reward_t_1'][self.experience_indices['t_1'][self.batch_indices]]
        done = self.experience['done'][self.experience_indices['t_1'][self.batch_indices]]

        return observation_t, action_t, observation_t_1, reward_t_1, done

    def _get_batch_indices(self, batch_size, fixed_batch_size, replace=True):
        transition_counter = self.experience_indices['t_1'].size
        sample_indices = np.arange(transition_counter)
        if fixed_batch_size and transition_counter == 0:
            self.batch_indices = np.zeros(batch_size, np.int)
        elif not fixed_batch_size and transition_counter == 0:
            self.batch_indices = np.zeros(1, np.int)
        elif not fixed_batch_size and transition_counter < batch_size:
            self.batch_indices = np.random.choice(sample_indices, transition_counter, replace=replace)
        else:
            self.batch_indices = np.random.choice(sample_indices, batch_size, replace=replace)

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
        # Initialise number filled counter
        n = 0
        while n < n_to_fill:
            observation_t = env.reset()
            done = False

            # Run environment for one episode
            while n < n_to_fill and not done:
                action_t = env.action_space.sample()
                observation_t_1, reward_t_1, done, _ = env.step(action_t)

                #display_observation(observation_t_1)
                self.add(observation_t, action_t, observation_t_1, reward_t_1, done)

                observation_t = observation_t_1
                n += 1

    def _add_to_experience(self, add_index, action_t, observation_t_1, reward_t_1, done):
        self.experience['action_t'][add_index] = action_t
        self.experience['observation_t_1'][add_index] = observation_t_1
        self.experience['reward_t_1'][add_index] = reward_t_1
        self.experience['done'][add_index] = done

    def _roll_experience_indices(self):
        for key in self.experience_indices.keys():
            self.experience_indices[key] = np.roll(self.experience_indices[key], -1)

    def _update_experience_indices(self):
        # TODO - comment
        if self.experience_indices['t_1'].size is not 0:
            if self.experience_indices['t_1'][-1] + 1 < self.experience['done'].size:
                self.next_done = self.experience['done'][self.experience_indices['t_1'][-1] + 1]
            else:
                self.next_done = self.experience['done'][0]

        if self.experience_indices['t_1'].size > 1:
            if self.experience_indices['t_1'][-1] + 2 < self.experience['done'].size:
                self.next_next_done = self.experience['done'][self.experience_indices['t_1'][-1] + 2]
            else:
                self.next_next_done = self.experience['done'][self.experience_indices['t_1'][-1] + 2 - self.experience['done'].size]

        # Precompute reused contitions
        filling_or_increasing_transitions = (self.filling or (self.next_done and not self.prev_done))
        reducing_transitions = ((not self.filling) and self.prev_done and not (self.next_done or self.next_next_done))

        if not filling_or_increasing_transitions:
            # Roll sample indices
            self._roll_experience_indices()

        if self.prev_done:
            # If the previous step was the end of an episode we need to increment 2 indices so that we can write
            # both the first and second step of the next episode (to make 1 full transition)
            index_increment = 2
        else:
            # Otherwise we just need to increment by 1
            index_increment = 1

        for key in self.experience_indices.keys():
            if self.experience_indices[key].size is 0:
                # Covers the case where we need to start filling empty arrays
                if key == 't':
                    new_index = 0
                else:
                    new_index = 1
            else:
                if filling_or_increasing_transitions:
                    # If we are still filling up the buffer or we will be overwriting a final step of another episode
                    # (i.e. reducing the number of stored episodes by 1) we want to be appending the indices so we get
                    # the previous index to increment from the end of the experience indices array
                    new_index = self.experience_indices[key][-1] + index_increment
                else:
                    # Otherwise we want to overwrite the final element in the experience indices array with the
                    # correct index
                    new_index = self.experience_indices[key][-2] + index_increment

                if new_index >= self.size:
                    # If we have incremented the index past the final index of the experience arrays we need to set it
                    # back to 0 or 1 depending on whether we are starting a new episode or not
                    new_index = new_index - self.size

            if filling_or_increasing_transitions:
                # If we are still filling up the memory, or we are overwriting the final step of a episode and not
                # recording the start of a new episode (so reducing the number of stored episodes) we can append to
                # experience indices rather than overwriting as we are increasing the number of stored transitions
                self.experience_indices[key] = np.append(self.experience_indices[key],
                                                         new_index)
            else:
                # Otherwise we need to just overwrite the final index (after rolling the index arrays)
                self.experience_indices[key][-1] = new_index

            if reducing_transitions:
                # If we are adding the start of a new episode that does not end after 2 timesteps and is not overwriting
                # the start of another stored episode (so increasing the number of stored episodes) we need to delete
                # the first index from experience indices as we have overwritten o_t of that transition
                self.experience_indices[key] = self.experience_indices[key][1:]


class PrioritisedExperienceReplayBufferM(ExperienceReplayBufferM):
    def __init__(self, size, observation_shape, alpha, observation_dtype=np.uint8, action_dtype=np.uint8,
                 reward_dtype=np.int16, done_dtype=np.uint8, td_error_dtype=np.float32):
        super(self.__class__, self).__init__(size, observation_shape,
                                                                 observation_dtype,
                                                                 action_dtype,
                                                                 reward_dtype,
                                                                 done_dtype)
        self.alpha = alpha
        self.eps = 1e-7
        self.max_priority = self.eps**self.alpha

        self.priority_sum = self.eps  # Initialising to eps so we don't accidentally divide by zero

        self.experience['priority'] = np.zeros(size, dtype=td_error_dtype) + self.eps

    def add(self, observation_t, action_t, observation_t_1, reward_t_1, done):
        super().add(observation_t, action_t, observation_t_1, reward_t_1, done)

        # Set priority of newly added experience to max priority
        # - this may be sped up by keeping track of max rather than recalculating
        self.experience['priority'][self.experience_indices['t_1'][-1]] = self.experience['priority'][self.experience_indices['t_1']].max()

        # Update sum of priorities
        #self.priority_sum = np.sum(self.experience['priority'][self.experience_indices['t_1']])

    def _get_batch_indices(self, batch_size, fixed_batch_size, replace=False):
        transition_counter = self.experience_indices['t_1'].size
        sample_indices = np.arange(transition_counter)
        if fixed_batch_size and transition_counter == 0:
            self.batch_indices = np.zeros(batch_size, np.int)
        elif not fixed_batch_size and transition_counter == 0:
            self.batch_indices = np.zeros(1, np.int)
        elif not fixed_batch_size and transition_counter < batch_size:
            self.batch_indices = np.random.choice(sample_indices, transition_counter, replace=replace,
                                                  p=self.experience['priority'][
                                                        self.experience_indices['t_1']] / np.sum(self.experience['priority'][self.experience_indices['t_1']]))
        else:
            self.batch_indices = np.random.choice(sample_indices, batch_size, replace=replace,
                                                  p=self.experience['priority'][
                                                        self.experience_indices['t_1']] / np.sum(self.experience['priority'][self.experience_indices['t_1']]))

    def update_priorities(self, batch_abs_td_error):
        """
        Updates the priorities of the last batch (the indices of which should now be stored in self.batch_indices) with
        the provided absolute td error for that batch. This should be called in a training loop after a batch has been
        sampled from the buffer and used for a training update
        :param batch_abs_td_error: absolute td error for the most recently sampled batch
        """
        # TODO - Look into using sum tree method mentioned in prioritised experience replay paper as speed up
        batch_priority = (batch_abs_td_error + self.eps) ** self.alpha

        # Add new TD errors to the correct place
        self.experience['priority'][self.experience_indices['t_1'][self.batch_indices]] = batch_priority

        # Update td error total sum
        #self.priority_sum = np.sum(self.experience['priority'][self.experience_indices['t_1']])


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

    def add(self,observation_t, action_t, observation_t_1, reward_t_1, done):
        """
        Add observation, action and subsequent reward to replay buffer
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
    #resized_observation = cv2.resize(cropped_observation, (84, 110))
    resized_observation = cv2.resize(cropped_observation, (64, 64))

    # Convert to correct dims array
    preprocessed_atari_observation = resized_observation
    return preprocessed_atari_observation


def plot_replay_buffer_data(experience, experience_indices, fig, ax, sample_indices=None):
    titles = ['Full Experience', 'Indexed Experience', 'Sampled Experience', 'Experience Indices', 'Sample Indices']
    for i, axes in enumerate(ax):
        axes.clear()
        axes.set_title(titles[i])

    # Combine experience into arrays
    full_experience = np.stack([experience['action_t'],
                                experience['observation_t_1'],
                                experience['reward_t_1'],
                                experience['done']], axis=1)

    indexed_experience = np.stack([experience['action_t'][experience_indices['t_1']],
                                   experience['observation_t_1'][experience_indices['t']],
                                   experience['observation_t_1'][experience_indices['t_1']],
                                   experience['reward_t_1'][experience_indices['t_1']],
                                   experience['done'][experience_indices['t_1']]], axis=1)

    sampled_experience = np.stack([experience['action_t'][experience_indices['t_1']],
                                   experience['observation_t_1'][experience_indices['t']],
                                   experience['observation_t_1'][experience_indices['t_1']],
                                   experience['reward_t_1'][experience_indices['t_1']],
                                   experience['done'][experience_indices['t_1']]], axis=1)

    full_indices = np.stack([experience_indices['t_1'],
                             experience_indices['t'],
                             experience_indices['t_1'],
                             experience_indices['t_1'],
                             experience_indices['t_1']], axis=1)

    sns.heatmap(full_experience, annot=True, ax=ax[0], fmt='d', cbar=False, xticklabels=['a_t', 'o_t_1', 'r_t_1', 'd_t_1'])
    sns.heatmap(indexed_experience, annot=True, ax=ax[1], fmt='d', cbar=False, xticklabels=['a_t', 'o_t', 'o_t_1', 'r_t_1', 'd_t_1'])
    sns.heatmap(sampled_experience, annot=True, ax=ax[2], fmt='d', cbar=False, xticklabels=['a_t', 'o_t', 'o_t_1', 'r_t_1', 'd_t_1'])
    sns.heatmap(full_indices, annot=True, ax=ax[3], fmt='d', cbar=False, xticklabels=['a_t', 'o_t', 'o_t_1', 'r_t_1', 'd_t_1'])

    plt.show(block=False)


def display_observation(observation):
    n_frames = observation.shape[2]
    fix, ax = plt.subplots(1, n_frames)
    for frame in range(n_frames):
        ax[frame].imshow(observation[:, :, frame])
    plt.show()


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



