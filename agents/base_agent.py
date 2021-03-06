import os
import copy
import json
import tensorflow as tf
import numpy as np
import gym
from agents import utils


class BaseAgent(object):
    # To build your agent, you only to pass a "configuration" which is a dictionary
    def __init__(self, config):
        # I like to keep the best HP found so far inside the model itself
        # This is a mechanism to load the best HP and override the configuration
        if config['best']:
            config.update(self.get_best_config())

        # I make a `deepcopy` of the configuration before using it
        # to avoid any potential mutation when I iterate asynchronously over configurations
        self.config = copy.deepcopy(config)

        if config['debug']:  # This is a personal check i like to do
            print('config', self.config)

        # When working with NN, one usually initialize randomly
        # and you want to be able to reproduce your initialization so make sure
        # you store the random seed and actually use it in your TF graph (tf.set_random_seed() for example)
        self.random_seed = self.config['random_seed']

        # All models share some basics hyper parameters, this is the section where we
        # copy them into the model
        self.result_dir = self.config['result_dir']
        self.test_result_dir = self.config['test_result_dir']
        self.max_iter = self.config['max_iter']
        self.max_train_episodes = self.config['max_train_episodes']
        self.test_episodes = self.config['test_episodes']
        self.test_every = self.config['test_every']
        self.render_test_every = self.config['render_test_every']
        self.render_every = self.config['render_every']
        self.drop_keep_prob = self.config['drop_keep_prob']
        self.learning_rate = self.config['learning_rate']
        self.l2 = self.config['l2']
        self.batch_size = self.config['batch_size']
        self.replay_buffer_size = self.config['replay_buffer_size']
        self.gamma = self.config['gamma']
        self.epsilon_init = self.config['epsilon']
        self.update_target_every = self.config['update_target_every']
        self.double_q = self.config['double_q']
        self.prioritised_replay = self.config['prioritised_replay']
        self.alpha = self.config['alpha']
        self.replay_buffer_init_fill = self.config['replay_buffer_init_fill']
        self.frames_to_stack = self.config['frames_to_stack']
        self.repeat_count = self.config['repeat_count']
        self.final_exploration_frame = self.config['final_exploration_frame']
        self.epsilon_final = self.config['epsilon_final']
        self.momentum = self.config['momentum']
        self.summary_every = self.config['summary_every']
        self.clip_rewards = self.config['clip_rewards']
        self.save_every = self.config['save_every']
        self.preprocessor_func_name = self.config['preprocessor_func']

        #  Other initialisations
        self.env_steps = 0
        self.train_iter = 0

        # Create environment
        self.preprocess_func = utils.select_preprocessor_func(self.preprocessor_func_name)  # TODO - Specific function not compatible with cartpole
        env = gym.make(self.config['env'])
        self.env = utils.EnvWrapper(env,
                                    preprocessor_func=self.preprocess_func,
                                    frames_to_stack=self.frames_to_stack,
                                    repeat_count=self.repeat_count,
                                    clip_rewards=self.clip_rewards)

        # Now the child Model needs some custom parameters, to avoid any
        # inheritance hell with the __init__ function, the model
        # will override this function completely
        self.set_model_props(config)

        # Set up global step
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Again, child Model should provide its own build_graph function
        self.graph = self.build_graph(self.graph)

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=10)
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            # Add test summary ops
            self.test_placeholders = {'mean_episode_length': tf.placeholder(tf.float32),
                                      'mean_episode_return': tf.placeholder(tf.float32)}
            mean_episode_length_summary = tf.summary.scalar('mean_episode_length', self.test_placeholders['mean_episode_length'])
            mean_episode_return_summary = tf.summary.scalar('mean_episode_return', self.test_placeholders['mean_episode_return'])

            self.test_summary = tf.summary.merge([mean_episode_length_summary, mean_episode_return_summary])

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        self.train_summary_writer = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(self.test_result_dir, self.sess.graph)

        # This function is not always common to all models, that's why it's again
        # separated from the __init__ one
        self.init()

    def set_model_props(self, config):
        # This function is here to be overridden completely.
        # When you look at your model, you want to know exactly which custom options it needs.
        pass

    def get_best_config(self):
        # This function is here to be overridden completely.
        # It returns a dictionary used to update the initial configuration (see __init__)
        return {}

    @staticmethod
    def get_random_config(fixed_params={}):
        # Why static? Because you want to be able to pass this function to other processes
        # so they can independently generate random configuration of the current model
        raise Exception('The get_random_config function must be overridden by the agent')

    def build_graph(self, graph):
        raise Exception('The build_graph function must be overridden by the agent')

    def act(self, observation):
        raise Exception('The act function must be overridden by the agent')

    def test(self):
        # Initialise performance record arrays
        episode_lengths = np.zeros(self.test_episodes)
        episode_returns = np.zeros(self.test_episodes)

        for episode in range(self.test_episodes):
            # Initialise episode
            observation = self.env.reset()
            done = False
            episode_length = 0
            episode_return = 0

            # Check whether to render this episode
            render = self.render_test_every > 0 and (episode % self.render_test_every) == 0

            while not done:
                # Render environment if required
                if render:
                    self.env.render()

                # Increment episode length
                episode_length += 1

                # Decide next action
                action = self.act(observation)

                # Act and get next observation, reward and episode status
                observation, reward, done, _ = self.env.step(action)

                # Increment total reward
                episode_return += reward

            # Record episode length and total reward
            episode_lengths[episode] = episode_length
            episode_returns[episode] = episode_return

        # Compute mean episode length and total reward
        mean_episode_length = np.mean(episode_lengths)
        mean_episode_return = np.mean(episode_returns)

        return mean_episode_length, mean_episode_return

    def learn_from_episode(self, learn_every=1, render=False):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overridden by the agent')

    def train(self):
        # This function is usually common to all your models
        for self.episode_id in range(0, self.max_train_episodes):
            if self.episode_id % self.render_every == 0:
                render = True
            else:
                render = False

            if self.episode_id % self.test_every == 0:
                mean_episode_length, mean_episode_return = self.test()
                test_feed_dict = {self.test_placeholders['mean_episode_length']: mean_episode_length,
                                  self.test_placeholders['mean_episode_return']: mean_episode_return}

                test_summary = self.sess.run(self.test_summary, feed_dict=test_feed_dict)
                self.test_summary_writer.add_summary(test_summary, self.env_steps)

            # Perform all TensorBoard operations within learn_from_episode
            self.learn_from_episode(render=render)

            # If you don't want to save during training, you can just pass a negative number
            if self.save_every > 0 and self.episode_id % self.save_every == 0:
                self.save()

    def save(self):
        # This function is usually common to all your models, Here is an example:
        if self.config['debug']:
            print('Saving to %s' % self.result_dir)
        self.saver.save(self.sess, self.result_dir + '/model-ep_' + str(self.episode_id))

        # I always keep the configuration that
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.sess.run(self.init_op)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)



