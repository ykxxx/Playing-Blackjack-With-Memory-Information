import tensorflow as tf
import numpy as np
from qnet import Qnet
from beliefnet import BeliefNetwork
import os
import collections


def one_hot(data, depth=10, num_deck=4):

    # arr = np.zeros([depth])
    arr = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 16])
    arr = arr * 4

    for d in data:
        arr[d-1] -= 1

    return arr


class Agent:
    def __init__(self, name, config, sess):

        self.name = name
        self.config = config
        self.training = self.config.training
        self.sess = sess
        self.memory_size = self.config.memory_size
        self.num_memory = self.config.num_memory
        self.agent_memory = []
        self.num_obv = self.config.num_obv
        self.num_action = self.config.num_action
        self.action_chosen = tf.placeholder(tf.int32, [None])
        self.action_one_hot = tf.one_hot(self.action_chosen, self.num_action)
        self.private_encoding_length = self.num_action
        self.fixed_context_length = [8, 6]
        self.fixed_single_length = [12, 20]

        # Q net
        with tf.variable_scope('Q_primary', reuse=tf.AUTO_REUSE):
            self.primary_model = Qnet(self.config, self.sess, 'Q_primary')
        with tf.variable_scope('Q_target', reuse=tf.AUTO_REUSE):
            self.target_model = Qnet(self.config, self.sess, 'Q_target')
        
        # Belief net
        with tf.variable_scope('Belief', reuse=tf.AUTO_REUSE):
            self.belief_model = BeliefNetwork(self.config, self.sess, self.memory_size)

        self.saver = tf.train.Saver(max_to_keep=None)

    def save_ckpt(self, sess, path, global_step):

        if not os.path.exists(path):
            os.makedirs(path)

        self.saver.save(sess, save_path=os.path.join(path, "checkpoint"), global_step=global_step)
        print('Saved <%d> ckpt to %s' % (global_step, path))

    def restore_ckpt(self, path, sess):
        ckpt_status = tf.train.get_checkpoint_state(path)
        if ckpt_status:
            self.saver.restore(sess, ckpt_status.model_checkpoint_path)
        if ckpt_status:
            print('Load model from %s' % path)
            return True
        print('Fail to load model from %s' % path)
        return False

    def choose_action(self, q_values, epsilon):

        if not self.training:
            maximums = np.argwhere(q_values == np.amax(q_values)).flatten().tolist()
            action_chosen = int(np.random.choice(maximums, 1))
        elif np.random.random() < epsilon:
            action_chosen = np.random.randint(0, self.num_action)
        else:
            if self.config.soft_max:
                expq = np.exp(self.config.acting_boltzman_beta * (np.squeeze(q_values) - np.max(q_values)))
                act_prob = expq / np.sum(expq)
                action_chosen = int(np.random.choice(self.num_action, 1, p=act_prob))
            else:
                action_chosen = np.argmax(q_values)

        action_one_hot = self.sess.run(self.action_one_hot, feed_dict={
            self.action_chosen: [action_chosen]
        })

        return action_one_hot, action_chosen

    def game_strategy(self, flipped_cards):

        if self.config.strategy[:1] == "m":

            if len(flipped_cards) >= self.memory_size:
                self.agent_memory = flipped_cards[-self.memory_size:]
            else:
                self.agent_memory = np.zeros([self.memory_size])
                self.agent_memory = flipped_cards

            self.agent_memory = one_hot(self.agent_memory, 10, self.config.num_deck)
        elif self.config.strategy[:1] == "c":

            target = int(self.config.strategy[1:])
            num_target = collections.Counter(flipped_cards)[target]
            self.agent_memory = np.expand_dims(num_target, axis=0)

        return self.agent_memory

    def learn(self, buffer):

        sample_idx = np.random.choice(buffer.size, self.config.batch_size, replace=False)
        samples = [buffer.get_data(i) for i in sample_idx]
        samples_length = [buffer.get_step(i) for i in sample_idx]
        max_length = np.max(samples_length)

        b_obv = np.zeros([self.config.batch_size, max_length, self.config.num_obv])
        b_memory = np.zeros([self.config.batch_size, max_length, self.config.num_memory])
        # b_hidden_states = np.zeros([self.config.batch_size, max_length, self.config.out_x_dim])
        b_action_chosen = np.zeros([self.config.batch_size, max_length, 1])
        b_reward = np.zeros([self.config.batch_size, max_length, 1])
        b_terminal = np.zeros([self.config.batch_size, max_length, 1])
        b_belief = np.zeros([self.config.batch_size, max_length, self.config.num_belief])
        b_workplace = np.zeros([self.config.batch_size, max_length, 10])
        b_new_card = np.zeros([self.config.batch_size, max_length, 10])

        for i in range(len(samples)):
            for j in range(len(samples[i])):
                b_obv[i][j] = samples[i][j][0]
                b_memory[i][j] = samples[i][j][1]
                # b_hidden_states[i][j] = samples[i][j][2]
                b_action_chosen[i][j] = samples[i][j][2]
                b_reward[i][j] = samples[i][j][3]
                b_terminal[i][j] = samples[i][j][4]
                b_belief[i][j] = samples[i][j][5]
                b_workplace[i][j] = samples[i][j][6]
                b_new_card[i][j] = samples[i][j][7]

        b_q_mask = np.zeros([self.config.batch_size, max_length, self.config.num_action])
        b_target_qs = np.zeros([self.config.batch_size, max_length])
        belief_mask = np.ndarray.astype(b_new_card, float)
        belief_target = np.zeros([self.config.batch_size, max_length])

        b_q_values = self.target_model.get_q_values(b_obv, b_belief)
        max_target_qs = np.max(b_q_values, axis=2)

        for j in reversed(range(max_length)):
            for i in range(len(samples)):
                action_index = int(b_action_chosen[i][j])
                b_q_mask[i, j, action_index] = 1 if j < len(samples[i]) else 0
                if int(b_terminal[i][j][0]) == 1:
                    b_target_qs[i][j] = b_reward[i][j][0]
                    belief_target[i][j] = 1
                elif j < len(samples[i]) - 1:
                    # vs = 0
                    # sample_len = len(samples[i]) - 1
                    # for k in range(j, sample_len):
                    #     vs += max_target_qs[i][k] * np.power(self.config.gamma, k - j)
                    # b_target_qs[i][j] = vs + b_reward[i][sample_len][0] * np.power(self.config.gamma, sample_len)
                    b_target_qs[i][j] = b_reward[i][j][0] + self.config.gamma * max_target_qs[i][j + 1]
                    belief_target[i][j] = 1

        loss = self.primary_model.fit_target_q(b_obv, b_belief, b_action_chosen[:, :, 0], b_target_qs, b_q_mask)
        belief_loss = self.belief_model.fit_target_belief(belief_target, belief_mask, b_obv, b_memory, b_workplace)

        return loss, belief_loss


def main():

    sess = tf.Session()

    exec ("from config import config", globals())

    agent = Agent("player", config, sess)
    flipped_cards = [5, 3, 7, 10, 10, 9, 8, 10, 10, 10, 10, 10, 3, 5, 8, 9, 3, 3, 2]
    agent_memory = agent.game_strategy(flipped_cards)
    print(agent_memory)


if __name__ == "__main__":
    main()
