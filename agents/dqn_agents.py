import tensorflow as tf
import numpy as np
from agents.base_agent import BaseAgent
from agents import utils
from agents import losses


class DQNAgent(BaseAgent):
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
        raise Exception('The get_random_config function must be overriden by the agent')

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.placeholders = {'observation_t': tf.placeholder('float', [None, ]),
                                 'action_t': tf.placeholder('int', [None]),
                                 'observation_t_1': tf.placeholder('float', [None, ]),
                                 'reward_t_1': tf.placeholder('float', [None]),
                                 'done': tf.placeholder('int', [None])}

            # Create Q t net
            q_t = []  # TODO - need to deal with parameter sharing between q_t and q_t_1

            # Create Q t + 1 net
            q_t_1 = []

            # Create op to get action based on observation t
            self.action = tf.argmax(q_t)

            self.loss = losses.one_step_td_loss(self.placeholders['reward_t_1'],
                                                self.gamma,
                                                q_t,
                                                q_t_1,
                                                self.placeholders['action_t'],
                                                self.placeholders['done'])

            optimiser = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimiser.minimize(self.loss)

            self.summary = tf.summary.merge_all()
        return graph

    def act(self, observation, explore=False):
        # If required choose action according to exploration policy
        if (np.random.uniform() < self.epsilon) and explore is True:
            action = self.env.action_space.sample()
        else:
            action = self.sess.run(self.action, feed_dict={self.placeholders['observation_t']: observation})
        return action

    def learn_from_episode(self, learn_every=1):
        observation_t = self.env.reset()
        done = False

        while not done:
            # Decide which action to take based on current observation
            action_t = self.act(observation_t)

            # Take action in environment and get next observation, reward and done bool
            observation_t_1, reward_t_1, done, _ = self.env.step(action_t)

            # Add experience to replay buffer
            self.replayBuffer.add(observation_t, action_t, observation_t_1, reward_t_1, done)

            # Learn from experience
            if self.env_steps % learn_every == 0:
                # Get batch of experience from replay buffer
                feed_dict = self.replayBuffer.get_batch_feed_dict(self.batch_size, self.placeholders)

                # Update parameters
                self.sess.run(self.train_op, feed_dict=feed_dict)

                # Perform Tensorboard operations
                # TODO - tensorboard operations

                self.train_iter += 1

            self.env_steps += 1

            # Set next observation to current observation for next iteration
            observation_t = observation_t_1

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overridden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.sess.run(self.init_op)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

        # Initialise experience replay buffer
        state_shape = self.env.observation_space.shape
        self.replayBuffer = utils.ExperienceReplayBuffer(self.replay_buffer_size, state_shape)
