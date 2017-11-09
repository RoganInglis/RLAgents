import tensorflow as tf
from agents import losses
import numpy as np
from agents import value_networks
from agents import utils
import gym
import time
from pympler import asizeof


def test_out(pass_condition, name):
    if pass_condition:
        print('Passed: ' + name)
        return True
    else:
        print('Failed: ' + name)
        return False


def one_step_td_loss_test():
    action_dim = 2
    batch_size = 64
    eps = 1e-9
    np.random.seed(1)

    reward_t_1 = tf.placeholder('float', [None])
    gamma = 0.99
    q_t = tf.placeholder('float', [None, action_dim])
    q_t_1 = tf.placeholder('float', [None, action_dim])
    action_t = tf.placeholder(tf.int32, [None])
    done = tf.placeholder('float', [None])

    loss = losses.one_step_td_loss(reward_t_1, gamma, q_t, q_t_1, action_t, done)

    q_target = tf.placeholder('float', [None])
    q_estimate = tf.placeholder('float', [None])

    correct_loss_op = tf.losses.huber_loss(q_target, q_estimate)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # Define actual values to use for test
    reward_t_1_val = np.random.uniform(-10, 10, batch_size)
    q_t_val = np.random.uniform(-100, 100, (batch_size, action_dim))
    q_t_1_val = np.random.uniform(-100, 100, (batch_size, action_dim))
    action_t_val = np.random.randint(0, 2, batch_size)
    done_val = np.random.randint(0, 2, batch_size).astype(np.float32)

    feed_dict = {reward_t_1: reward_t_1_val,
                 q_t: q_t_val,
                 q_t_1: q_t_1_val,
                 action_t: action_t_val,
                 done: done_val}

    masked_max_q = (1.0 - done_val) * np.max(q_t_1_val, axis=1)

    q_target_val = reward_t_1_val + gamma * masked_max_q
    q_estimate_val = q_t_val[np.arange(batch_size), action_t_val]

    q_feed_dict = {q_target: q_target_val,
                   q_estimate: q_estimate_val}

    # Get loss as computed by one_step_td_loss function
    loss_val = sess.run(loss, feed_dict)

    # Get loss as computed here
    loss_true_val = sess.run(correct_loss_op, q_feed_dict)

    return test_out(np.all(np.abs(loss_true_val - loss_val) < eps), 'one_step_td_loss test')


def get_scope_variables_test():
    var_1 = tf.Variable('float', [None], name='var_1')
    name = 'scope_1'

    with tf.variable_scope(name) as scope:
        var_2 = tf.Variable('int', [None, 10], name='var_2')
        var_3 = tf.Variable('float', [None, 5], name='var_3')

    vars = scope.trainable_variables()

    return test_out(vars == [var_2, var_3], 'get_scope_variables_test')


def function_pass_to_function_test():
    def function_to_pass(x, y, add=True):
        if add:
            return x + y
        else:
            return x * y

    def function_to_pass_to(func, args):
        return func(*args, False)

    x = 2
    y = 5
    args = [x, y]

    z = function_to_pass_to(function_to_pass, args)

    return test_out(z == x * y, 'function_pass_to_function_test')


def conv_out_size_test():
    input_shape = [84, 84]
    padding = 'same'
    kernel_shape = 8
    strides = 4
    out_shape = value_networks.conv_out_size(input_shape, padding, kernel_shape, strides)

    true_out_shape = [int((84 + 7 - 8) / 4), int((84 + 7 - 8) / 4)]

    return test_out(out_shape == true_out_shape, 'conv_out_size_test')


def dqn_atari_conv_net_test():
    x = tf.placeholder('float', [None, 84, 84, 4])
    y1 = value_networks.dqn_atari_conv_net(x, 4)
    y2 = value_networks.dqn_atari_conv_net(x, 4, reuse=True)

    result_dir = 'results\\test\\dqn_atari_conv_net_test'

    sess = tf.Session()
    tf.summary.FileWriter(result_dir, sess.graph)

    print('dqn_atari_conv_net graph saved in ' + result_dir + ' for viewing using Tensorboard')


def frame_buffer_test():
    eps = 1e-9
    # Create frame buffer
    frameBuffer = utils.FrameBuffer([2, 2], 4)

    # Define some arrays to add to the frame buffer
    add_1 = np.array([[1.0, 1.1], [1.2, 1.3]])
    add_2 = add_1 + 1
    add_3 = add_2 + 1
    add_4 = add_3 + 1
    add_5 = add_4 + 1

    # Add arrays to frame buffer
    frameBuffer.add(add_1)
    frameBuffer.add(add_2)
    frameBuffer.add(add_3)
    frameBuffer.add(add_4)
    frameBuffer.add(add_5)

    # Define correct result of retrieving from frame buffer
    correct_result = np.stack([add_2, add_3, add_4, add_5], axis=2)

    result = frameBuffer.get()
    correct_array = np.abs(correct_result - result) > eps

    frameBuffer.reset()

    test_out(correct_array.all(), 'frame_buffer_test')


def experience_replay_buffer_test():
    # Create experience replay buffer
    env = gym.make('MsPacman-v0')
    obs = env.reset()
    preprocessed_obs = utils.preprocess_atari(obs)

    replayBuffer = utils.ExperienceReplayBuffer(100, [110, 84, 4])

    replayBuffer.fill_with_random_experience(replayBuffer.size, env, utils.preprocess_atari)

    batch = replayBuffer.get_batch(32)

    obs = env.reset()


def preprocess_atari_test():
    pass


def env_wrapper_test():
    env = gym.make('MsPacman-v0')
    preprocessor_func = utils.preprocess_atari
    frames_to_stack = 4
    repeat_count = 4
    env = utils.EnvWrapper(env=env,
                           preprocessor_func=preprocessor_func,
                           frames_to_stack=frames_to_stack,
                           repeat_count=repeat_count)

    observation_t = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        observation_t_1, reward_t_1, done, _ = env.step(action)

        observation_t = observation_t_1


class TestEnv:
    def __init__(self, min_episode_steps=1, max_episode_steps=5, observation_shape=None, observation_dtype=np.uint8):
        self.min_episode_steps = min_episode_steps
        self.max_episode_steps = max_episode_steps
        self.episode_step = 0
        self.total_step = 0
        self.done = False
        self.episode_length = np.random.randint(self.min_episode_steps, self.max_episode_steps + 1)
        self.action_space = TestActionSpace(self.episode_step)
        self.observation_shape = observation_shape
        self.observation_dtype = observation_dtype

    def step(self, action):
        # Increment to next episode step
        self.episode_step += 1
        self.total_step += 1
        self.action_space.episode_step = self.total_step

        # Set observation and reward as episode step
        reward = self.total_step
        if self.observation_shape is None:
            observation = self.total_step
        else:
            observation = np.zeros(self.observation_shape, self.observation_dtype)

        # Check whether the episode is complete
        if self.episode_step == self.episode_length:
            self.done = True

        info = None
        return observation, reward, self.done, info

    def reset(self):
        self.episode_step = 0
        if self.total_step is not 0:
            self.total_step += 1
        self.action_space.episode_step = self.total_step
        self.done = False
        self.episode_length = np.random.randint(self.min_episode_steps, self.max_episode_steps + 1)

        if self.observation_shape is None:
            observation = self.total_step
        else:
            observation = np.zeros(self.observation_shape, self.observation_dtype)
        return observation


class TestActionSpace:
    def __init__(self, init_episode_step=0):
        self.episode_step = init_episode_step

    def sample(self):
        return self.episode_step


def test_env_test():
    env = TestEnv(1, 5)
    n_episodes = 3

    for episode in range(n_episodes):
        observation_t = env.reset()
        action = 0
        done = False

        while not done:
            observation_t_1, reward_t_1, done, _ = env.step(action)

            print("o_t: {}, a_t: {}, o_t_1: {}, r_t_1: {}, done: {}".format(observation_t,
                                                                            action,
                                                                            observation_t_1,
                                                                            reward_t_1,
                                                                            done))

            observation_t = observation_t_1
            action += 1


def new_experience_replay_buffer_test():
    replayBuffer = utils.ExperienceReplayBufferM(100, [], observation_dtype=np.int32,
                                                 action_dtype=np.int32, reward_dtype=np.int32)
    env = TestEnv()

    replayBuffer.fill_with_random_experience(2000, env)

    print_batch(*replayBuffer.get_batch(32))
    print('test')


def print_batch(observation_t, action_t, observation_t_1, reward_t_1, done):
    for index in range(observation_t.size):
        print("o_t: {}, a_t: {}, o_t_1: {}, r_t_1: {}, done: {},  Correct numbers: {}".format(
            observation_t[index],
            action_t[index],
            observation_t_1[index],
            reward_t_1[index],
            done[index],
            (observation_t[index] == action_t[index]) and (observation_t_1[index] == reward_t_1[index]) and (observation_t_1[index] - observation_t[index] == 1)))


def new_experience_replay_buffer_timing_test():
    observation_shape = [84, 84, 4]
    replayBufferM = utils.ExperienceReplayBufferM(100, observation_shape, reward_dtype=np.int32, action_dtype=np.int32)
    replayBuffer = utils.ExperienceReplayBuffer(100, observation_shape)
    envM = TestEnv(observation_shape=observation_shape)
    env = TestEnv(observation_shape=observation_shape)

    n_repeats = 10000

    time_m_add = time_add_to_replay_buffer(envM, replayBufferM, n_repeats)
    time_add = time_add_to_replay_buffer(env, replayBuffer, n_repeats)

    print("BufferM add time: {:10.9f}s, Buffer old add time: {:10.9f}s, Ratio: {:4.3f}".format(time_m_add, time_add,
                                                                                               time_m_add / time_add))

    # n_repeats = 100

    time_m_get = time_get_batch_from_replay_buffer(envM, replayBufferM, n_repeats)
    time_get = time_get_batch_from_replay_buffer(env, replayBuffer, n_repeats)

    print("BufferM get time: {:10.9f}s, Buffer old get time: {:10.9f}s, Ratio: {:4.3f}".format(time_m_get, time_get,
                                                                                               time_m_get / time_get))


def time_add_to_replay_buffer(env, replayBuffer, n_repeats):
    n_added = 0
    times = np.array([])
    while n_added < n_repeats:
        observation_t = env.reset()
        done = False
        while not done and (n_added < n_repeats):
            action_t = env.action_space.sample()
            observation_t_1, reward_t_1, done, _ = env.step(action_t)

            start_time = time.time()
            replayBuffer.add(observation_t, action_t, observation_t_1, reward_t_1, done)
            times = np.append(times, time.time() - start_time)

            n_added += 1
            observation_t = observation_t_1

    mean_time = np.mean(times)
    return mean_time


def time_get_batch_from_replay_buffer(env, replayBuffer, n_repeats, batch_size=32):
    n_batches = 0
    times = np.array([])
    # Fill buffer
    replayBuffer.fill_with_random_experience(replayBuffer.size, env)

    while n_batches < n_repeats:
        start_time = time.time()
        batch = replayBuffer.get_batch(batch_size)
        times = np.append(times, time.time() - start_time)

        n_batches += 1

    mean_time = np.mean(times)
    return mean_time


def replay_buffer_size_test():
    replayBufferM = utils.ExperienceReplayBufferM(1000, [84, 84, 4])
    replayBuffer = utils.ExperienceReplayBuffer(1000, [84, 84, 4])

    replay_buffer_m_size = asizeof.asizeof(replayBufferM)
    replay_buffer_size = asizeof.asizeof(replayBuffer)

    print(
        "Replay buffer M size: {} bytes, Replay buffer old size: {} bytes, Ratio: {:4.3f}".format(replay_buffer_m_size,
                                                                                                  replay_buffer_size,
                                                                                                  replay_buffer_m_size / replay_buffer_size))


if __name__ == '__main__':
    # one_step_td_loss_test()
    get_scope_variables_test()
    function_pass_to_function_test()
    conv_out_size_test()
    # dqn_atari_conv_net_test()
    # frame_buffer_test()
    # experience_replay_buffer_test()
    # env_wrapper_test()
    # test_env_test()
    new_experience_replay_buffer_test()
    new_experience_replay_buffer_timing_test()
    replay_buffer_size_test()
