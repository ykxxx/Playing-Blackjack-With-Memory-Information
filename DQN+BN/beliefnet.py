import tensorflow as tf
import numpy as np


class BeliefNetwork:
    def __init__(self, config, sess, memory_size):
        self.config = config
        self.sess = sess
        self.belief_len = 10
        self.memory_size = memory_size
        self.num_obv = self.config.num_obv
        self.num_memory = self.config.num_memory
        self.num_action = self.config.num_action
        self.decay_step = self.config.decay_step
        self.decay_rate = self.config.decay_rate

        self.global_step = tf.Variable(0, trainable=False)

        self.layer_dims = [64, 32]
        # Batch_Size X Sequence_len X belief_dim X num_actions
        # self.init_belief = [1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 4/13]

        self.obv = tf.placeholder(tf.float32, [None, None, self.num_obv])
        self.memory_obv = tf.placeholder(tf.float32, [None, None, self.belief_len])
        self.workplace = tf.placeholder(tf.float32, [None, None, self.belief_len])
        # self.belief = tf.placeholder(tf.float32, [None, None, self.belief_len])
        self.num_card_played = tf.expand_dims(self.obv[:, :, -1], axis=-1)
        self.num_card_remember = tf.expand_dims(tf.reduce_sum(self.memory_obv, axis=-1), axis=-1)
        self.belief_old = self.memory_obv / self.num_card_remember
        self.uncertainty = self.num_card_played - self.num_card_remember

        self.input = tf.concat([self.workplace, self.belief_old, self.uncertainty], axis=-1)
        self.layers = [self.input]

        for dim in self.layer_dims:
            kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=1e-2)
            self.layers.append(tf.layers.dense(self.layers[-1], dim, kernel_initializer=kernel_initializer))

        # self.layers.append(tf.layers.conv1d(self.layers[-1], filters=self.num_action, kernel_size=1,
        #     strides=1, padding='VALID', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-2)))

        self.layers.append(tf.layers.dense(self.layers[-1], self.config.num_belief, activation=tf.nn.softmax))
        # self.likelihood = self.layers[-1] + 1e-9
        self.belief_new = self.layers[-1]

        self.belief_target = tf.placeholder(tf.float32, [None, None])
        self.belief_mask = tf.placeholder(tf.float32, [None, None, self.config.num_belief])

        self.belief_loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.belief_new * self.belief_mask, axis=2)
                                             - self.belief_target)) / tf.reduce_sum(self.belief_mask)

        self.belief_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
                                v.name.startswith('Belief')]

        if self.config.decay_lr:
            self.learning_rate = tf.train.exponential_decay(self.config.belief_learning_rate, self.global_step,
                                                            self.decay_step, self.decay_rate, staircase=True)
        else:
            self.learning_rate = self.config.belief_learning_rate

        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.belief_opt = self.optimizer.minimize(self.belief_loss, var_list=self.belief_varlist_)

    def fit_target_belief(self, belief_target, belief_mask, obv, memory_obv, workplace):
        loss, _ = self.sess.run([self.belief_loss, self.belief_opt], feed_dict={
            self.belief_target: belief_target,
            self.belief_mask: belief_mask,
            self.obv: obv,
            self.memory_obv: memory_obv,
            self.workplace: workplace
        })

        return loss

    def get_belief(self, obv, workplace, memory_obv):

        belief = self.sess.run(self.belief_new, feed_dict={
            self.workplace: workplace,
            self.memory_obv: memory_obv,
            self.obv: obv
        })

        return belief


def main():
    from easydict import EasyDict as edict
    config = edict({'memory_dim': 20, 'fc_layer_info': [30, 20], 'num_actions': 10})
    B_net = BeliefNetwork(config, tf.random.uniform(shape=[4, 1, 4, 10]), tf.random.uniform(shape=[4, 5, 10]),
                          tf.random.uniform(shape=[4, 5, 10]), tf.random.uniform(shape=[4, 1, 4]))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    [ib, likelihood, nb] = sess.run([B_net.initial_belief_, B_net.likelihood_, B_net.new_belief_],
                                    feed_dict={B_net.memory_obv_input_: np.zeros((4, 20))})
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()