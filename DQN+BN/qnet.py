import tensorflow as tf
import numpy as np
from gru import GRU
import os


class Qnet:
    def __init__(self, config, sess, scope_name):

        self.config = config
        self.sess = sess
        self.memory_size = self.config.memory_size
        self.num_memory = self.config.num_memory
        self.num_obv = self.config.num_obv
        self.num_action = self.config.num_action
        # self.hidden_dim = self.config.hidden_dim
        # self.in_x_dim = self.config.in_x_dim
        # self.out_x_dim = self.config.out_x_dim
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma
        self.decay_step = self.config.decay_step
        self.decay_rate = self.config.decay_rate
        self.scope_name = scope_name

        self.global_step = tf.Variable(0, trainable=False)

        # with tf.variable_scope(self.scope_name):

        self.obv = tf.placeholder(tf.float32, [None, None, self.num_obv], name="obv")
        self.memory = tf.placeholder(tf.float32, [None, None, self.num_memory], name="memory")
        self.actions = tf.placeholder(tf.int32, [None, None, self.num_action])
        self.action_chosen = tf.placeholder(tf.int32, [None, None])
        self.belief = tf.placeholder(tf.float32, [None, None, self.config.num_belief])

        # RNN
        # self.gru = GRU(self.hidden_dim, self.in_x_dim, self.out_x_dim, self.config.gpu, self.sess)
        # self.in_x = tf.placeholder(tf.float32, [None, None, self.in_x_dim])
        # self.in_state = tf.placeholder(tf.float32, [None, self.hidden_dim])
        # self.hidden_states = tf.placeholder(tf.float32, [None, None, self.out_x_dim])
        # self.out_states = tf.placeholder(tf.float32, [None, self.hidden_dim])

        # Qnet
        # self.input = tf.concat([self.obv, self.memory, self.hidden_states], axis=-1)
        self.input = tf.concat([self.obv, self.belief], axis=-1)
        self.layers = [self.input]
        for i in range(config.num_layer - 1):
            activation = tf.nn.relu
            initializer = None  # tf.keras.initializers.he_normal()
            num_nodes = config.scale_factor * (config.num_layer - i - 1)

            self.layers.append(
                tf.layers.dense(self.layers[-1], num_nodes, activation=activation, kernel_initializer=initializer))
            # self.layers.append(tf.layers.batch_normalization(self.layers[-1], training=config.training))

            # num_nodes = round(np.power((self.num_action / config.q_input_dim), (i - 1) / (config.num_layer - 1)))

        self.layers.append(tf.layers.dense(self.layers[-1], self.num_action, activation=tf.nn.softmax))
        self.q_values = self.layers[-1]
        self.q_mask = tf.placeholder(tf.float32, [None, None, self.num_action])
        self.target_q = tf.placeholder(tf.float32, [None, None])

        self.loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.q_values * self.q_mask, axis=2)
                                             - self.target_q)) / tf.reduce_sum(self.q_mask)

        self.q_primary_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) \
                                   if v.name.startswith('Q_primary')]
        self.q_target_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) \
                                  if v.name.startswith('Q_target')]
        self.q_target_update_ = tf.group([v_t.assign(v_t * (1 - 0.2) + v * 0.2) \
                                          for v_t, v in zip(self.q_target_varlist_, self.q_primary_varlist_)])
        if self.config.decay_lr:
            self.learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step,
                                                            self.decay_step, self.decay_rate, staircase=True)
        else:
            self.learning_rate = self.config.learning_rate

        self.opt_q_ = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # self.q_training_op_ = self.opt_q_.minimize(self.loss, var_list=self.q_primary_varlist_)

        # prediction = tf.reduce_sum(self.q_values * self.q_mask, axis=2)
        # self.loss = tf.losses.huber_loss(labels=self.target_q, predictions=prediction)

    # def get_rnn_hidden_state(self, obv, action, in_state):
    #
    #     action = np.expand_dims(action, axis=1)
    #     in_x = np.concatenate([obv, action], axis=-1)
    #
    #     hidden_states, out_states = self.gru(in_x, in_state)
    #
    #     return hidden_states, out_states

    def get_q_values(self, obv, belief):

        q_values = self.sess.run(self.q_values, feed_dict={
            self.obv: obv,
            self.belief: belief,
            # self.hidden_states: hidden_states
        })

        return q_values

    def fit_target_q(self, obv, belief, action_chosen, target_q, q_mask):

        q_values = self.get_q_values(obv, belief)

        q_pred = np.sum(q_values * q_mask, axis=2)

        optimizer, loss = self.sess.run([self.opt_q_, self.loss], feed_dict={
            self.obv: obv,
            self.belief: belief,
            # self.hidden_states: hidden_states,
            self.action_chosen: action_chosen,
            self.target_q: target_q,
            self.q_mask: q_mask
        })

        return loss

    def update_target_net(self):

        self.sess.run(self.q_target_update_)


def main():

    sess = tf.Session

    # model = Qnet(config, sess)


if __name__ == "__main__":
    main()
