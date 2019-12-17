import numpy as np
import tensorflow as tf
import os
import sys
import logging
from agent import Agent
from blackjack import Blackjack
from replay_buffer import ReplayBuffer


def play(player, game, buffer, epsilon, config):

    game.start()

    terminal = False
    step = 0
    trajectory = []
    # hidden_states = np.zeros(([1, 1, config.in_x_dim]))
    # in_states = np.zeros([1, config.hidden_dim])

    while not terminal:
        obv = game.workspace
        obv_e = np.expand_dims(np.expand_dims(obv, axis=0), axis=0)
        player_cards = len(game.player)
        dealer_cards = len(game.dealer) - 1
        card_left = 208 - len(game.flipped_cards)
        flipped_cards = game.flipped_cards[:-1 * (player_cards + dealer_cards)]
        memory = player.game_strategy(flipped_cards)
        memory = np.concatenate([memory, [player.memory_size, card_left]], axis=-1)
        memory_e = np.expand_dims(np.expand_dims(memory, axis=0), axis=0)
        q_values = player.primary_model.get_q_values(obv_e, memory_e)
        action_one_hot, action_chosen = player.choose_action(q_values, epsilon)
        game = game.proceed(action_chosen)
        reward = game.reward
        terminal = game.terminate

        trajectory.append([obv, memory, action_chosen, reward, terminal])
        step += 1

    total_rewards = game.game_rewards
    buffer.store(trajectory, step, game.result, game.blackjack)
    return trajectory, total_rewards, game.result, game.num_steps


def main():

    exp_path = sys.argv[1]
    # exp_path = "player_m170"
    save_path = os.path.join("Experiments", exp_path)

    exec('from Experiments.%s.config import config' % exp_path, globals())

    config.player_name = exp_path
    config.strategy = input("player strategy:")
    config.memory_size = int(input("player memory size: "))
    # config.num_memory = int(input("memory dimension: "))

    logger = logging.getLogger()
    logger.handlers = []
    file_handler = logging.FileHandler(os.path.join(save_path, "Log"))
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    sess = tf.Session()

    player = Agent(config.player_name, config, sess)
    game = Blackjack(config.num_deck, config.success_reward, config.fail_reward, config.step_reward, 0)
    buffer = ReplayBuffer(config)
    rewards = []
    losses = []
    results = []
    steps = []

    init = tf.global_variables_initializer()
    sess.run(init)

    player.restore_ckpt(os.path.join(save_path, "CKPT"), sess)

    print("start training %s with config in %s" % (config.player_name, save_path))

    for episode in range(int(config.start_iteration) + 1, int(config.end_iteration) + 1):

        epsilon_decay = config.switch_iters / (int(episode / config.switch_iters) * 2 + 3)
        
        epsilon = config.epsilon_min + (config.epsilon_start - config.epsilon_min) \
                  * np.exp(-1 * (episode % config.switch_iters) / epsilon_decay)

        # epsilon_start = config.epsilon_start
        # epsilon_min = config.epsilon_min

        # if episode % config.switch_iters == 0:
        #     epsilon_start *= 0.7
        #     epsilon_min *= 0.7

        # epsilon = config.epsilon_min + (config.epsilon_start - config.epsilon_min) \
        #           * np.exp(-1 * (episode % config.switch_iters) / config.epsilon_decay)

        epsilon = 1 if episode < 1000 else epsilon

        traj, reward, result, step = play(player, game, buffer, epsilon, config)
        rewards.append(reward)
        results.append(result)
        steps.append(step)

        if buffer.size >= config.batch_size:
            loss = player.learn(buffer)
            losses.append(loss)

        if episode % config.update_q_target_frequency == 0:
            player.target_model.update_target_net()

        if episode % config.print_iteration == 0:
            s_rate, f_rate, t_rate, avg_step, b_rate = buffer.get_performance()
            logging.info("[{0:d}] epsilon: {1:.3f}, average reward: {2:.3f}, average loss: {3:.3f}, "
                         "success rate: {4:.3f}, fail rate: {5:.3f}, tie rate: {6:.3f}, average step: {7:.3f}, "
                         "blackjack rate: {8:.3f}"
                  .format(episode, epsilon, np.mean(rewards), np.mean(losses), s_rate, f_rate, t_rate, np.mean(steps), b_rate))

            rewards = []
            losses = []
            results = []

        if episode % config.save_iteration == 0:
            player.save_ckpt(sess, os.path.join(save_path, "CKPT"), global_step=episode)

    player.save_ckpt(sess, os.path.join(save_path, "CKPT"), global_step=int(config.end_iteration))

    return


if __name__ == "__main__":
    main()