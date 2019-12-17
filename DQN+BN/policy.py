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
            game.reset()
            game.start()
            game.player = player_input
            game.dealer[-1] = dealer_input[0]
            obv = game.obv
            obv_e = np.expand_dims(np.expand_dims(obv, axis=0), axis=0)
            workplace = game.workplace
            workplace_e = np.expand_dims(np.expand_dims(workplace, axis=0), axis=0)
            flipped_cards = game.flipped_cards
            memory_obv = player.game_strategy(flipped_cards)
            memory_obv_e = np.expand_dims(np.expand_dims(memory_obv, axis=0), axis=0)
            # memory_e = np.expand_dims(np.expand_dims(memory, axis=0), axis=0)
            belief = player.belief_model.get_belief(obv_e, workplace_e, memory_obv_e)
            q_values = player.primary_model.get_q_values(obv_e, belief)
            action_one_hot, action_chosen = player.choose_action(q_values, 0)
            f.write("%d, " % int(action_chosen))
        f.write("\n")







    return


if __name__ == "__main__":
    main()