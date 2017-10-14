import tensorflow as tf


def multi_layer_perceptron(input_tensor, neurons_per_layer, activations=tf.nn.relu):
    x = input_tensor
    if type(activations) is list:
        for n, activation in zip(neurons_per_layer, activations):
            x = tf.layers.dense(x, n, activation)
    else:
        for n in neurons_per_layer:
            x = tf.layers.dense(x, n, activations)
    return x
