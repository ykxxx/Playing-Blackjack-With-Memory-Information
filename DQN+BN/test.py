import numpy as np
import tensorflow as tf
import os
from agent import Agent
from blackjack import Blackjack
from replay_buffer import ReplayBuffer
from train import play
from tqdm import tqdm
import sys


def sample_game(traj):
    # obv, memory, action_chosen, reward, terminal
    for i in range(len(traj)):
        print("cards:   1 2 3 4 5 6 7 8 9 10\n")
        print("player: ", traj[i][0][0:11])
        print("dealer: ", traj[i][0][11:21])
        print("memory: ", traj[i][1])
        print("action chosen: ", traj[i][2])

def main():

    exp_path = sys.argv[1]
    save_path = os.path.join("Experiments", exp_path)
    ckpt_path = os.path.join(save_path, "CKPT")
    exec('from Experiments.%s.config import config' % exp_path, globals())
    config.soft_max = False

    if not os.path.isdir(save_path):
        print('Cannot find target folder')
        exit()

    sess = tf.Session()

    player = Agent(config.player_name, config, sess)
    game = Blackjack(config.num_deck, config.success_reward, config.fail_reward, config.step_reward, config.memory_size)
    buffer = ReplayBuffer(config)
    rewards = []

    player.restore_ckpt(path=ckpt_path, sess=sess)
    count = 0

    for episode in tqdm(range(1, int(1000))):

        epsilon = 0
        traj, reward, result = play(player, game, buffer, epsilon, config)
        rewards.append(reward)

        if result == 1 and (traj[-1][2] == 1 or traj[-1][2] == 2):
            count += 1

    s_rate, f_rate, t_rate, avg_step, b_rate, double, surrender = buffer.get_performance()
    f = open(os.path.join(save_path, "result.txt"), "w+")
    f.write("average reward: {0:.3f}, success rate: {1:.3f}, fail rate: {2:.3f}, "
          "tie rate: {3:.3f}, average step: {4:.3f}, blackjack rate: {5:.3f}, double: {6:.3f}, surrender: {7:.3f}"
            .format(np.mean(rewards), s_rate, f_rate, t_rate, avg_step, b_rate, double, surrender))
    print("average reward: {0:.3f}, success rate: {1:.3f}, fail rate: {2:.3f}, "
          "tie rate: {3:.3f}, average step: {4:.3f}, blackjack rate: {5:.3f}, double: {6:.3f}, surrender: {7:.3f}"
            .format(np.mean(rewards), s_rate, f_rate, t_rate, avg_step, b_rate, double, surrender))

    print("count:", count)

    return


if __name__ == "__main__":
    main()