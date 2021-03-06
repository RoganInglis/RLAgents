import tensorflow as tf


def build_q_nets(q_func, q_func_args_list, scope='q_net', double_q=False):
    # Build networks and get trainable variables
    q_t, q_t_vars = q_func(*q_func_args_list[0], scope=scope)
    q_t_1, q_t_1_vars = q_func(*q_func_args_list[1], scope=scope + '_1')

    # If using double q learning we need an additional network for Qt+1 that shares the same parameters as the Qt net
    # but takes observation t+1 as input rather then observation t
    if double_q:
        q_t_1_d, _ = q_func(*q_func_args_list[1], scope=scope, reuse=True)
    else:
        q_t_1_d = None

    # Add tensorboard ops
    tf.summary.histogram('q_t', q_t)
    tf.summary.histogram('q_t_1', q_t_1)
    tf.summary.scalar('mean_max_q_t', tf.reduce_mean(tf.reduce_max(q_t, axis=1), axis=0))
    tf.summary.scalar('mean_max_q_t_1', tf.reduce_mean(tf.reduce_max(q_t_1, axis=1), axis=0))

    # Add update ops and group to allow copying of online net variables to target net
    assign_ops = []
    with tf.name_scope('target_update_ops'):
        for q_t_var, q_t_1_var in zip(q_t_vars, q_t_1_vars):
            assign_ops.append(tf.assign(q_t_1_var, q_t_var))
            tf.summary.histogram(q_t_var.name, q_t_var)
            tf.summary.histogram(q_t_1_var.name, q_t_1_var)
        q_t_1_update_op = tf.group(*assign_ops)

    return q_t, q_t_1, q_t_1_update_op, q_t_1_d


def multi_layer_perceptron(input_tensor, neurons_per_layer, activations=tf.nn.relu, scope='mlp', reuse=None):
    with tf.variable_scope(scope, reuse=reuse) as net_scope:
        x = input_tensor

        # If the input tensor is not 2D we need to split and stack it so that it is
        len_x_shape = len(x.shape)
        while len_x_shape > 2:
            # Unstack along the final axis
            x = tf.unstack(x, axis=len_x_shape - 1)

            # Concatenate along the new final axis
            x = tf.concat(x, axis=len_x_shape - 2)

            # Recompute shape
            len_x_shape = len(x.shape)

        name_count = 0
        if type(activations) is list:
            for n, activation in zip(neurons_per_layer, activations):
                x = tf.layers.dense(x, n, activation, name=str(name_count), reuse=reuse)
                name_count += 1
        else:
            for n in neurons_per_layer[:-1]:
                x = tf.layers.dense(x, n, activations, name=str(name_count), reuse=reuse)
                name_count += 1
            x = tf.layers.dense(x, neurons_per_layer[-1], name=str(name_count), reuse=reuse, use_bias=False)

        return x, net_scope.trainable_variables()


def dqn_atari_conv_net(input_tensor, n_actions, scope='atari_conv_net', reuse=None, padding='valid'):
    with tf.variable_scope(scope, reuse=reuse) as net_scope:
        x = tf.layers.conv2d(input_tensor, 32, 8, 4, padding, activation=tf.nn.relu, name='conv_1', reuse=reuse)
        x = tf.layers.conv2d(x, 64, 4, 2, padding, activation=tf.nn.relu, name='conv_2', reuse=reuse)
        x = tf.layers.conv2d(x, 64, 3, 1, padding, activation=tf.nn.relu, name='conv_3', reuse=reuse)
        x = tf.layers.dense(tf.contrib.layers.flatten(x), 512, tf.nn.relu, name='dense_1', reuse=reuse)
        x = tf.layers.dense(x, n_actions, name='dense_2', reuse=reuse)

        return x, net_scope.trainable_variables()


def conv_out_size(input_shape, padding, kernel_shape, strides):
    # Convert int kernel shape and strides to 2d if required
    if type(kernel_shape) is int:
        kernel_shape = [kernel_shape, kernel_shape]
    if type(strides) is int:
        strides = [strides, strides]

    if padding == 'same':
        p = [(kernel_shape[0] - 1)/2, (kernel_shape[0] - 1)/2]
    else:
        p = [0, 0]

    out_size = [int((input_shape[0] + 2*p[0] - kernel_shape[0])/strides[0]),
                int((input_shape[1] + 2*p[1] - kernel_shape[1])/strides[1])]
    return out_size


if __name__ == '__main__':
    x = tf.placeholder('float', [None, 84, 84, 4])
    y1 = multi_layer_perceptron(x, [100, 80, 60, 40, 20])
    y2 = multi_layer_perceptron(x, [100, 80, 60, 40, 20], reuse=True)

    graph = tf.Graph()
    saver = tf.train.Saver(max_to_keep=50)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    result_dir = 'results\\test'

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=sess_config)
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

    sess.run(init_op)