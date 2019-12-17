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
        obv = game.obv
        workplace = game.workplace
        new_card = game.new_card
        obv_e = np.expand_dims(np.expand_dims(obv, axis=0), axis=0)
        workplace_e = np.expand_dims(np.expand_dims(workplace, axis=0), axis=0)
        # player_cards = len(game.player)
        # dealer_cards = len(game.dealer) - 1
        # card_left = 208 - len(game.flipped_cards)
        flipped_cards = game.flipped_cards
        memory_obv = player.game_strategy(flipped_cards)
        memory_obv_e = np.expand_dims(np.expand_dims(memory_obv, axis=0), axis=0)
        # memory_e = np.expand_dims(np.expand_dims(memory, axis=0), axis=0)
        belief = player.belief_model.get_belief(obv_e, workplace_e, memory_obv_e)
        q_values = player.primary_model.get_q_values(obv_e, belief)
        action_one_hot, action_chosen = player.choose_action(q_values, epsilon)
        game = game.proceed(action_chosen)
        reward = game.reward
        terminal = game.terminate

        trajectory.append([obv, memory_obv, action_chosen, reward, terminal, belief, workplace, new_card])
        step += 1

    total_rewards = game.game_rewards
    buffer.store(trajectory, step, game.result, game.blackjack)
    return trajectory, total_rewards, game.result, game.num_steps


def main():
    exp_path = sys.argv[1]
    # exp_path = "player_m20"
    save_path = os.path.join("Experiments", exp_path)

    exec('from Experiments.%s.config import config' % exp_path, globals())

    config.player_name = exp_path
    config.strategy = input("player strategy:")
    config.memory_size = int(input("player memory size: "))
    game_init_len = int(input("game initial state: "))
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
    game = Blackjack(config.num_deck, config.success_reward, config.fail_reward, config.step_reward, game_init_len)
    buffer = ReplayBuffer(config)
    rewards = []
    losses = []
    b_losses = []
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

        epsilon = 1 if episode < 500 else epsilon

        traj, reward, result, step = play(player, game, buffer, epsilon, config)
        rewards.append(reward)
        results.append(result)
        steps.append(step)

        if buffer.size >= config.batch_size:
            loss, b_loss = player.learn(buffer)
            losses.append(loss)
            b_losses.append(b_loss)

        if episode % config.update_q_target_frequency == 0:
            player.target_model.update_target_net()

        if episode % config.print_iteration == 0:
            s_rate, f_rate, t_rate, avg_step, b_rate = buffer.get_performance()
            logging.info("[{0:d}] epsilon: {1:.3f}, average reward: {2:.3f}, average loss: {3:.3f}, average b_loss: {4:.3f} "
                         "success rate: {5:.3f}, fail rate: {6:.3f}, tie rate: {7:.3f}, average step: {8:.3f}, "
                         "blackjack rate: {9:.3f}"
                         .format(episode, epsilon, np.mean(rewards), np.mean(losses), np.mean(b_losses), s_rate, f_rate, t_rate,
                                 np.mean(steps), b_rate))

            rewards = []
            losses = []
            results = []

        if episode % config.save_iteration == 0:
            player.save_ckpt(sess, os.path.join(save_path, "CKPT"), global_step=episode)

    player.save_ckpt(sess, os.path.join(save_path, "CKPT"), global_step=int(config.end_iteration))

    return


if __name__ == "__main__":
    main()