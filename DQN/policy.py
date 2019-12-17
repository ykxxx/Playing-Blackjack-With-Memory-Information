import sys
import os
import numpy as np
import tensorflow as tf
from agent import Agent
from blackjack import Blackjack


def main():

    exp_path = sys.argv[1]
    save_path = os.path.join("Experiments", exp_path)
    ckpt_path = os.path.join(save_path, "CKPT")
    exec('from Experiments.%s.config import config' % exp_path, globals())
    config.soft_max = False
    config.training = False
    f = open(os.path.join(save_path, "policy.csv"), "w+" )

    if not os.path.isdir(save_path):
        print('Cannot find target folder')
        exit()

    sess = tf.Session()

    player = Agent(config.player_name, config, sess)
    player.restore_ckpt(path=ckpt_path, sess=sess)
    game = Blackjack(config.num_deck, config.success_reward, config.fail_reward, config.step_reward, 0)

    dealer_inputs = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    player_inputs = []
    f.write("player/dealer, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10\n")

    for i in range(1, 11):
        for j in range(1, 11):
            player_hand = [i, j]
            player_inputs.append(player_hand)

    for player_input in player_inputs:
        f.write("%s, " % player_input)
        for dealer_input in dealer_inputs:
            game.start()
            game.player = player_input
            game.dealer[-1] = dealer_input[0]
            game.get_workspace(False)
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
            action_one_hot, action_chosen = player.choose_action(q_values, 0)
            f.write("%d, " % int(action_chosen))
        f.write("\n")







    return


if __name__ == "__main__":
    main()