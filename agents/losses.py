import tensorflow as tf


def one_step_td_loss(reward_t_1, gamma, q_t, q_t_1, action_t, done, double_q=False, q_t_1_d=None):
    with tf.name_scope('one_step_td_loss'):
        with tf.name_scope('Q_Estimate'):
            # Get shape of action vector (required for indexing in next step)
            action_shape = tf.shape(action_t)

            # Get action index array to be used to index q_t array to get action value of that experience
            action_t_index = tf.transpose(tf.stack([tf.range(action_shape[0]), action_t]))
            value_estimate = tf.gather_nd(q_t, action_t_index)

        # Define target value, using stop_gradient as this should not contribute to the gradient update and
        # dealing with the fact that the value after the terminal state should be 0 by multiplying with (1 - done)
        # where done is an indicator representing whether the current state is terminal, stored as 1 for terminal, 0 not
        with tf.name_scope('Target'):
            if double_q and (q_t_1_d is not None):
                q_t_1_d_shape = tf.shape(q_t_1_d)
                q_t_1_index = tf.transpose(tf.stack([tf.range(q_t_1_d_shape[0]), tf.argmax(q_t_1_d, axis=1,
                                                                                           output_type=tf.int32)]))
                max_q = tf.gather_nd(q_t_1, q_t_1_index)
            else:
                max_q = tf.reduce_max(q_t_1, axis=1)
            next_q = tf.multiply((1.0 - done), max_q)
            value_target = tf.stop_gradient(reward_t_1 + gamma*next_q)

            tf.summary.histogram('next_q', next_q)

        # Compute mean Huber loss for batch
        loss = tf.losses.huber_loss(value_target, value_estimate)
    return loss
