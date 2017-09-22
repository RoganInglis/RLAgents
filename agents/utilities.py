import numpy as np


class ExperienceReplayBuffer:
    # TODO comment
    def __init__(self, size, state_shape, action_shape):
        self.size = size

        # Make sure shapes are lists
        state_shape = list(state_shape)
        action_shape = list(action_shape)

        # Get replay shapes
        state_replay_shape = state_shape.insert(0, size)
        action_replay_shape = action_shape.insert(0, size)

        # Initialise experience with zeros
        self.experience = {'s_t': np.zeros(state_replay_shape),
                           'a_t': np.zeros(action_replay_shape),
                           'r_t_1': np.zeros(size)}

        # Initialise fill counter and sample indices
        self.fill_counter = 0
        self.sample_indices = np.array([])

    def add(self, s_t, a_t, r_t_1):
        """
        Add state, action and subsequent reward to replay buffer
        :param s_t: state at time t
        :param a_t: action at time t
        :param r_t_1: reward at time t + 1
        """
        # Determine index at which to add experience
        if self.fill_counter < self.size:
            #
            self.sample_indices = np.append(self.sample_indices, self.fill_counter)
            add_index = self.sample_indices[-1]

            self.fill_counter += 1
        else:
            # Roll sample indices
            self.sample_indices = np.roll(self.sample_indices, -1)
            add_index = self.sample_indices[-1]

        self.experience['s_t'][add_index] = s_t
        self.experience['a_t'][add_index] = a_t
        self.experience['r_t_1'][add_index] = r_t_1

    def get_batch(self, batch_size, fixed_batch_size=False):
        # TODO comment
        # Get batch indices
        if fixed_batch_size and self.fill_counter == 0:
            batch_indices = np.zeros(batch_size, np.int)
        elif not fixed_batch_size and self.fill_counter == 0:
            batch_indices = np.zeros(1, np.int)
        elif not fixed_batch_size and self.fill_counter - 1 < batch_size:
            batch_indices = np.random.choice(self.sample_indices[:-1], self.fill_counter)
        else:
            batch_indices = np.random.choice(self.sample_indices[:-1], batch_size)

        # Sample from replay buffer
        s_t = self.experience['s_t'][batch_indices]
        a_t = self.experience['a_t'][batch_indices]
        s_t_1 = self.experience['s_t_1'][batch_indices + 1]
        r_t_1 = self.experience['r_t_1'][batch_indices]

        return s_t, a_t, s_t_1, r_t_1

    def get_batch_feed_dict(self, batch_size, placeholders_dict, fixed_batch_size=False):
        # TODO comment
        # Get batch
        s_t, a_t, s_t_1, r_t_1 = self.get_batch(batch_size, fixed_batch_size)

        # Convert to feed dict
        feed_dict = {placeholders_dict['s_t']: s_t,
                     placeholders_dict['a_t']: a_t,
                     placeholders_dict['s_t_1']: s_t_1,
                     placeholders_dict['r_t_1']: r_t_1}

        return feed_dict




