import os
import json
import time
import tensorflow as tf
import numpy as np
from agents import value_networks

# See the __init__ script in the models folder
# `make_model` is a helper function to load any models you have
from agents import make_agent

# I personally always like to make my paths absolute
# to be independent from where the python binary is called
project_dir = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags


# Hyper-parameters search configuration
flags.DEFINE_boolean('fullsearch', False, 'Perform a full search of hyperparameter space ex:(hyperband -> lr search -> hyperband with best lr)')
flags.DEFINE_boolean('dry_run', False, 'Perform a dry_run (testing purpose)')
flags.DEFINE_integer('nb_process', 4, 'Number of parallel process to perform a HP search')

# fixed_params is a trick I use to be able to fix some parameters inside the model random function
# For example, one might want to explore different models fixing the learning rate,
# see the basic_model get_random_config function
flags.DEFINE_string('fixed_params', "{}", 'JSON inputs to fix some params in a HP search, ex: \'{"lr": 0.001}\'')

# Model configuration
flags.DEFINE_string('agent_name', 'DQNAgent', 'Unique name of the agent')
flags.DEFINE_boolean('best', False, 'Force to use the best known configuration')
flags.DEFINE_float('learning_rate', 2.5e-4, 'The learning rate of SGD')
flags.DEFINE_float('momentum', 0.95, 'The momentum of SGD')
flags.DEFINE_float('drop_keep_prob', 1.0, 'The dropout keep probability')
flags.DEFINE_float('l2', 0.0, 'L2 regularisation strength')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('replay_buffer_size', 200000, 'Number of timesteps to store in the replay buffer')
flags.DEFINE_integer('replay_buffer_init_fill', 50000, 'Number of random environment steps to initially store in the replay buffer')
flags.DEFINE_float('gamma', 0.99, 'Discount parameter for TD learning')
flags.DEFINE_float('epsilon', 1, 'Exploration parameter (initial value) for epsilon greedy exploration')
flags.DEFINE_float('epsilon_final', 0.1, 'Exploration parameter final value for decaying epsilon greedy')
flags.DEFINE_integer('final_exploration_frame', 1000000, 'Frame at which epsilon stops decaying')
flags.DEFINE_integer('update_target_every', 10000, 'Frequency at which to update the target q network')
flags.DEFINE_boolean('double_q', True, 'Use double Q learning')
flags.DEFINE_boolean('prioritised_replay', True, 'Use a prioritised replay buffer')
flags.DEFINE_float('alpha', 1, 'Prioritised replay strength parameter')
flags.DEFINE_integer('frames_to_stack', 4, 'Number of frames to stack as an observation')
flags.DEFINE_integer('repeat_count', 4, 'Number of frames to repeat each action for')
flags.DEFINE_boolean('clip_rewards', True, 'Clip_rewards to be -1, 0, 1')


# Environment configuration
flags.DEFINE_string('env', 'Boxing-v0', 'Name of the gym environment to use')

# Training configuration
flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_integer('max_iter', 100000000, 'Max number of training iterations')
flags.DEFINE_integer('max_train_episodes', 100000, 'Max number of training episodes')
flags.DEFINE_boolean('test', False, 'Load a model and compute test performance')
flags.DEFINE_integer('test_episodes', 10, 'Number of episodes over which to compute test results')
flags.DEFINE_integer('test_every', 30, 'Episode interval at which to test the agent during training')
flags.DEFINE_integer('render_test_every', 10000, 'Episode interval at which to render the environment during testing')
flags.DEFINE_integer('render_every', 500000, 'Episode interval at which to render the environment during training')
flags.DEFINE_integer('summary_every', 50, 'Interval at which to record a summary to tensorboard')
flags.DEFINE_integer('save_every', 10, 'Episode interval at which to save the model')

# This is very important for TensorBoard
# each model will end up in its own unique folder using time module
# Obviously one can also choose to name the output folder
flags.DEFINE_string('result_dir', project_dir + '/results/' + flags.FLAGS.agent_name + '/' + flags.FLAGS.env + '/' +
                    str(int(time.time())),
                    'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')
flags.DEFINE_string('test_result_dir', project_dir + '/results/' + flags.FLAGS.agent_name + '/' + flags.FLAGS.env + '/' +
                    str(int(time.time())) + '/test',
                    'Name of the directory to store/log the model test results (for TensorBoard)')

# Another important point, you must provide an access to the random seed
# to be able to fully reproduce an experiment
flags.DEFINE_integer('random_seed', np.random.randint(0, 2**8), 'Value of random seed')


def main(_):
    config = flags.FLAGS.__flags.copy()

    # fixed_params must be a string to be passed in the shell, let's use JSON
    config["fixed_params"] = json.loads(config["fixed_params"])

    if config['fullsearch']:
        print('Hyperparameter search not implemented yet')
    else:
        agent = make_agent(config)

        if config['test']:
            # Some code for testing ...
            agent.test()
        else:
            # Some code for training ...
            agent.train()


if __name__ == '__main__':
    tf.app.run()

