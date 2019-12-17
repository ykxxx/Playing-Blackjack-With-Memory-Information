import numpy as np
import tensorflow as tf
import os
from agent import Agent
from blackjack import Blackjack
from replay_buffer import ReplayBuffer
from train import play
from tqdm import tqdm
import sys


def main():

    exp_path = sys.argv[1]
    save_path = os.path.join("Experiments", exp_path)
    ckpt_path = os.path.join(save_path, "CKPT")
    exec('from Experiments.%s.config import config' % exp_path, globals())

    if not os.path.isdir(save_path):
        print('Cannot find target folder')
        exit()

    f = open(os.path.join(save_path, "results.txt"), "w+")
    g = open(os.path.join(save_path, "payout.csv"), "w+")

    sess = tf.Session()

    player = Agent(config.player_name, config, sess)
    game = Blackjack(config.num_deck, config.success_reward, config.fail_reward, config.step_reward, 0)
    buffer = ReplayBuffer(config)
    rewards = []
    steps = []

    ckpt_interval = int((config.end_iteration - config.start_iteration) / config.save_iteration)
    ckpt_iters = [config.save_iteration * i + config.start_iteration for i in range(1, ckpt_interval + 1)]
    config.soft_max = False
    config.training = False
    np.random.seed(1234)

    for ckpt_iter in ckpt_iters:

        w = open(os.path.join(ckpt_path, 'checkpoint'), 'w+')
        w.write('model_checkpoint_path: "checkpoint-%d"\nall_model_checkpoint_paths: "checkpoint-%d"' % (ckpt_iter, ckpt_iter))
        w.close()
        player.restore_ckpt(path=ckpt_path, sess=sess)

        for episode in tqdm(range(1, int(2000))):

            epsilon = 0
            traj, reward, result, step = play(player, game, buffer, epsilon, config)
            rewards.append(reward)
            steps.append(step)

        s_rate, f_rate, t_rate, avg_step, b_rate = buffer.get_performance()
        f.write("average reward: {0:.3f}, success rate: {1:.3f}, fail rate: {2:.3f}, tie rate: {3:.3f}, "
                "average step: {4:.3f}, blackjack rate: {5:.3f}\n"
                .format(np.mean(rewards), s_rate, f_rate, t_rate, np.mean(steps), b_rate))
        print("average reward: {0:.3f}, success rate: {1:.3f}, fail rate: {2:.3f}, tie rate: {3:.3f}, "
                "average step: {4:.3f}, blackjack rate: {5:.3f}\n"
                .format(np.mean(rewards), s_rate, f_rate, t_rate, np.mean(steps), b_rate))
        g.write("%f\n" % np.mean(rewards))

    return


if __name__ == "__main__":
    main()