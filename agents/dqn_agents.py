import tensorflow as tf
from agents.base_agent import BaseAgent
from agents import utilities


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
        raise Exception('The build_graph function must be overriden by the agent')

    def act(self, observation):
        raise Exception('The act function must be overriden by the agent')

    def learn_from_episode(self):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overriden by the agent')

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
        self.replay_buffer = utilities.ExperienceReplayBuffer(self.replay_buffer_size,
                                                              state_shape, [self.env.action_space.n])
