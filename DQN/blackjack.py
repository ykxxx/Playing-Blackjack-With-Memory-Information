import numpy as np
import random
import collections


def one_hot(data, depth=10):

    arr = np.zeros([depth])

    for d in data:
        arr[d-1] += 1

    return arr


class Blackjack():

    def __init__(self, num_deck, success_reward, fail_reward, step_reward, memory_size):

        self.num_deck = num_deck
        self.success_reward = success_reward
        self.fail_reward = fail_reward
        self.step_reward = step_reward
        self.memory_size = memory_size
        # self.players = players
        self.reward = 0
        self.game_rewards = 0
        self.num_steps = 0
        self.action_space = [0, 1]
        self.cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.deck = []
        self.flipped_cards = []
        # random.shuffle(self.deck)
        self.terminate = False
        self.bust = False
        self.blackjack = False
        self.result = None # win: 1, tie: 0, loss: -1
        # self.observation = None
        self.workspace = None
        self.double = False
        self.surrender = False
        self.first_act = True

    def get_new_deck(self):

        deck = self.num_deck * self.cards
        random.shuffle(deck)

        self.deck = deck[self.memory_size:]
        self.flipped_cards = deck[:self.memory_size]

    def deck_out(self):

        return len(self.deck) < self.num_deck * len(self.cards) - 180

    def draw_card(self):

        card = int(self.deck.pop())

        return card

    def useable_ace(self, cards):

        useable_ace = (collections.Counter(cards)[1] == 1) and (np.sum(cards) + 10 <= 21)

        return useable_ace

    def sum_cards(self, cards):

        sum = np.sum(cards) + 10 if self.useable_ace(cards) else np.sum(cards)

        return sum

    def is_bust(self, cards):

        return self.sum_cards(cards) > 21

    # def get_observation(self, terminate=False):
    #
    #     if terminate:
    #         obv = [tuple(sorted(self.player)), self.dealer, self.useable_ace(self.player), self.bust, self.blackjack]
    #     else:
    #         obv = [tuple(sorted(self.player[1:])), self.dealer[:1], self.useable_ace(self.player), self.bust]
    #     self.observation = obv

    def get_workspace(self, terminate=False):

        player = one_hot(self.player)
        player_sum = self.sum_cards(self.player)

        dealer = one_hot(self.dealer[1:])
        dealer_sum = self.sum_cards(self.dealer[1:])
        card_left = 208 - len(self.flipped_cards)

        if terminate is True:
            dealer = one_hot(self.dealer)
            dealer_sum = self.sum_cards(self.dealer)

        self.workspace = np.concatenate([[player_sum, 21 - player_sum, dealer_sum, 21 - dealer_sum, int(self.first_act), int(self.useable_ace(self.player))], player, dealer], axis=-1)

    def get_rewards(self, action):

        if self.terminate:
            dealer = self.sum_cards(self.dealer)
            player = self.sum_cards(self.player)
            self.blackjack = (player == 21)
            has_blackjack = 2 if (player == 21 or dealer == 21) else 1
            step = self.step_reward if (action == 1 or action == 2) else 0

            if (dealer == player) or (dealer > 21 and self.bust):
                self.result = 0
            elif (dealer < player or dealer > 21) and not self.bust:
                self.result = 1
            else:
                self.result = -1

            self.reward = step + self.result * has_blackjack * self.success_reward
            self.game_rewards = self.result * has_blackjack * self.success_reward

            # if self.double:
            #     self.reward *= 2
            #     self.game_rewards *= 2
            #
            # if self.surrender:
            #     self.reward = self.fail_reward / 2
            #     self.game_rewards = self.fail_reward / 2
            #     self.result = -1

        else:
            self.reward = self.step_reward

    def reset(self):

        # self.observation = None
        self.workspace = None
        self.reward = 0
        self.game_rewards = 0
        self.num_steps = 0
        self.terminate = False
        self.bust = False
        self.result = None
        self.blackjack = False
        self.flipped_cards = []
        self.double = False
        self.surrender = False
        self.first_act = True

        self.get_new_deck()

    def start(self):

        self.workspace = None
        self.reward = 0
        self.game_rewards = 0
        self.num_steps = 0
        self.terminate = False
        self.bust = False
        self.result = None
        self.blackjack = False
        self.double = False
        self.surrender = False
        self.first_act = True

        if self.deck_out():
            self.get_new_deck()

        self.player = [self.draw_card(), self.draw_card()]
        self.dealer = [self.draw_card(), self.draw_card()]
        self.get_workspace()
        self.flipped_cards.append(self.player[0])
        self.flipped_cards.append(self.player[-1])
        self.flipped_cards.append(self.dealer[-1])

        return self

    def proceed(self, action):

        # action: 0 - stand, 1 - hit

        if self.deck_out():
            self.get_new_deck()

        if action == 0:
            self.terminate = True

        elif action == 1:
            self.player.append(self.draw_card())
            self.get_workspace()
            self.flipped_cards.append((self.player[-1]))
            self.num_steps += 1

            if self.is_bust(self.player):
                self.terminate = True
                self.bust = True

        #     if action == 2:
        #         self.double = True
        #         self.terminate = True
        #
        # elif action == 3:
        #     self.surrender = True
        #     self.terminate = True

        if self.sum_cards(self.player) == 21:
            self.blackjack = True
            self.terminate = True

        if self.terminate:

            if not self.surrender or self.bust:

                while self.sum_cards(self.dealer) < 17:
                    self.dealer.append(self.draw_card())
                    self.flipped_cards.append(self.dealer[-1])

            self.get_workspace(True)
            self.flipped_cards.append(self.dealer[0])

        self.first_act = False
        self.get_rewards(action)

        return self


def main():

    memory_size = input("input memory size for blackjack game: ")

    blackjack = Blackjack(4, 1, -1, 0, int(memory_size))

    for i in range(10):
        print("round %d started" % i)
        blackjack.start()
        while not blackjack.terminate:
            print("first act:", blackjack.first_act)
            print("flipped cards:", blackjack.flipped_cards)
            print("player: ", blackjack.player)
            print("dealer: ", blackjack.dealer[1:])
            action = int(input("Enter player action, 0 for stand, 1 for hit: "))
            blackjack.proceed(action)
            print("reward:", blackjack.reward)
        print("round %d ended" % i)
        print("player:", blackjack.player)
        print("dealer:", blackjack.dealer)
        print("workspace:", blackjack.workspace)
        print("game rewards:", blackjack.game_rewards)
        print("busted:", blackjack.bust)
        print("blackjack:", blackjack.blackjack)
        print("result:", blackjack.result)


if __name__ == "__main__":
    main()









