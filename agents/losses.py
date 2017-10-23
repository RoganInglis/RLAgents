import tensorflow as tf


def one_step_td_loss(reward_t_1, gamma, q_t, q_t_1, action_t, done, debug_tb_ops=False):
    with tf.name_scope('one_step_td_loss'):
        # Get shape of action vector (required for indexing in next step)
        action_shape = tf.shape(action_t)

        # Get action index array to be used to index q_t array to get action value of that experience
        action_t_index = tf.transpose(tf.stack([tf.range(action_shape[0]), action_t]))
        value_estimate = tf.gather_nd(q_t, action_t_index)

        # Define target value, using stop_gradient as this should not contribute to the gradient update and
        # dealing with the fact that the value after the terminal state should be 0 by multiplying with (1 - done)
        # where done is an indicator representing whether the current state is terminal, stored as 1 for terminal, 0 not
        value_target = tf.stop_gradient(reward_t_1 + gamma*tf.multiply((1.0 - done), tf.reduce_max(q_t_1, axis=1)))

        if debug_tb_ops:
            tf.summary.histogram('next_q', tf.multiply((1 - done), gamma*tf.reduce_max(q_t_1)))

        # Compute mean Huber loss for batch
        loss = tf.losses.huber_loss(value_target, value_estimate)
    return loss
