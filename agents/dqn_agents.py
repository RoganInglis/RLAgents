import tensorflow as tf
import numpy as np
from agents.base_agent import BaseAgent
from agents import utils
from agents import losses
from agents import value_networks


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

            observation_placeholder_shape = [None]
            observation_placeholder_shape.extend(self.env.observation_space.shape)

            self.placeholders = {'observation_t': tf.placeholder('float', observation_placeholder_shape,
                                                                 name='observation_t'),
                                 'action_t': tf.placeholder(tf.int32, [None], name='action_t'),
                                 'observation_t_1': tf.placeholder('float', observation_placeholder_shape,
                                                                   name='observation_t_1'),
                                 'reward_t_1': tf.placeholder('float', [None], name='reward_t_1'),
                                 'done': tf.placeholder('float', [None], name='done')}

            # Create Q nets
            """
            q_func = value_networks.dqn_atari_conv_net
            q_func_args_list = [[self.placeholders['observation_t'], self.env.action_space.n],
                                [self.placeholders['observation_t_1'], self.env.action_space.n]]
            """

            q_func = value_networks.multi_layer_perceptron
            q_func_args_list = [[self.placeholders['observation_t'], [100, self.env.action_space.n]],
                                [self.placeholders['observation_t_1'], [100, self.env.action_space.n]]]

            q_t, q_t_1, self.target_update_op, q_t_1_d = value_networks.build_q_nets(q_func,
                                                                                     q_func_args_list,
                                                                                     double_q=self.double_q)

            # Create op to get action based on observation t
            self.action = tf.argmax(q_t, axis=1)
            tf.summary.histogram('action', self.action)

            # TODO - implement optional use of importance sampling correction for prioritised replay in loss function
            self.loss, td_error = losses.one_step_td_loss(self.placeholders['reward_t_1'],
                                                          self.gamma,
                                                          q_t,
                                                          q_t_1,
                                                          self.placeholders['action_t'],
                                                          self.placeholders['done'],
                                                          double_q=self.double_q,
                                                          q_t_1_d=q_t_1_d)

            self.abs_td_error = tf.abs(td_error)

            if self.l2 != 0.0:
                self.loss = self.loss + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'q_net_1' not in v.name]) * self.l2

            self.epsilon = tf.train.polynomial_decay(self.epsilon_init, self.global_step, self.final_exploration_frame,
                                                     self.epsilon_final)

            tf.summary.scalar('Loss', self.loss)

            optimiser = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
            self.train_op = optimiser.minimize(self.loss, global_step=self.global_step)

            self.train_summary = tf.summary.merge_all()
        return graph

    def act(self, observation, explore=False):
        # If required choose action according to exploration policy
        epsilon = self.sess.run(self.epsilon)
        if (np.random.uniform() < epsilon) and explore is True:
            action = self.env.action_space.sample()
        else:
            action = self.sess.run(self.action, feed_dict={self.placeholders['observation_t']: np.expand_dims(observation, 0)})
            action = action[0]
        return action

    def learn_from_episode(self, learn_every=1, render=False):
        observation_t = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            # Decide which action to take based on current observation
            action_t = self.act(observation_t, explore=True)

            # Take action in environment and get next observation, reward and done bool
            observation_t_1, reward_t_1, done, _ = self.env.step(action_t)

            # Add experience to replay buffer
            self.replayBuffer.add(observation_t, action_t, observation_t_1, reward_t_1, done)

            # Learn from experience
            if self.env_steps % learn_every == 0:
                # Get batch of experience from replay buffer
                feed_dict = self.replayBuffer.get_batch_feed_dict(self.batch_size, self.placeholders)

                # Update parameters
                ops = [self.train_op]
                if self.prioritised_replay:
                    ops.append(self.abs_td_error)
                if self.train_iter % self.update_target_every == 0:
                    ops.append(self.target_update_op)
                if self.env_steps % self.summary_every == 0:
                    ops.append(self.train_summary)
                train_out = self.sess.run(ops, feed_dict=feed_dict)

                # If using prioritised replay we need to update the sampling probabilities now
                if self.prioritised_replay:
                    self.replayBuffer.update_priorities(train_out[1])

                # Perform Tensorboard operations
                if self.env_steps % self.summary_every == 0:
                    self.train_summary_writer.add_summary(train_out[-1], self.env_steps)

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
        if self.prioritised_replay:
            self.replayBuffer = utils.PrioritisedExperienceReplayBufferM(self.replay_buffer_size, state_shape,
                                                                         alpha=self.alpha, reward_dtype=np.float32)
        else:
            self.replayBuffer = utils.ExperienceReplayBufferM(self.replay_buffer_size, state_shape,
                                                              reward_dtype=np.float32)
        self.replayBuffer.fill_with_random_experience(self.replay_buffer_init_fill, self.env)


class DQNAgentULoss(DQNAgent):
    def set_model_props(self, config):
        # This function is here to be overridden completely.
        # When you look at your model, you want to know exactly which custom options it needs.
        pass

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            observation_placeholder_shape = [None]
            observation_placeholder_shape.extend(self.env.observation_space.shape)

            self.placeholders = {'observation_t': tf.placeholder('float', observation_placeholder_shape,
                                                                 name='observation_t'),
                                 'action_t': tf.placeholder(tf.int32, [None], name='action_t'),
                                 'observation_t_1': tf.placeholder('float', observation_placeholder_shape,
                                                                   name='observation_t_1'),
                                 'reward_t_1': tf.placeholder('float', [None], name='reward_t_1'),
                                 'done': tf.placeholder('float', [None], name='done')}

            # Create Q nets
            q_func = value_networks.dqn_atari_conv_net
            q_func_args_list = [[self.placeholders['observation_t'], self.env.action_space.n],
                                [self.placeholders['observation_t_1'], self.env.action_space.n]]

            q_t, q_t_1, self.target_update_op, q_t_1_d = value_networks.build_q_nets(q_func,
                                                                                     q_func_args_list,
                                                                                     double_q=self.double_q)

            # Create op to get action based on observation t
            self.action = tf.argmax(q_t, axis=1)
            tf.summary.histogram('action', self.action)

            # TODO - implement optional use of importance sampling correction for prioritised replay in loss function
            self.loss, td_error = losses.one_step_td_loss(self.placeholders['reward_t_1'],
                                                          self.gamma,
                                                          q_t,
                                                          q_t_1,
                                                          self.placeholders['action_t'],
                                                          self.placeholders['done'],
                                                          double_q=self.double_q,
                                                          q_t_1_d=q_t_1_d)

            self.abs_td_error = tf.abs(td_error)

            if self.l2 != 0.0:
                self.loss = self.loss + tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'q_net_1' not in v.name]) * self.l2

            self.epsilon = tf.train.polynomial_decay(self.epsilon_init, self.global_step, self.final_exploration_frame,
                                                     self.epsilon_final)

            tf.summary.scalar('Loss', self.loss)

            optimiser = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
            self.train_op = optimiser.minimize(self.loss, global_step=self.global_step)

            self.train_summary = tf.summary.merge_all()
        return graph


