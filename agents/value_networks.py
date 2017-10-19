import tensorflow as tf


def multi_layer_perceptron(input_tensor, neurons_per_layer, activations=tf.nn.relu, scope='mlp', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        x = input_tensor
        name_count = 0
        if type(activations) is list:
            for n, activation in zip(neurons_per_layer, activations):
                x = tf.layers.dense(x, n, activation, name=str(name_count), reuse=reuse)
                name_count += 1
        else:
            for n in neurons_per_layer:
                x = tf.layers.dense(x, n, activations, name=str(name_count), reuse=reuse)
                name_count += 1
        return x


if __name__ == '__main__':
    x = tf.placeholder('float', [None, 10])
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