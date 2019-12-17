import numpy as np


class ReplayBuffer:

    def __init__(self, config):

        self.max_batch = config.max_batch
        self.size = 0
        assert self.size <= self.max_batch

        self.storage = []
        self.result = []
        self.step = []
        self.blackjack = []
        self.double = []
        self.surrender = []

    def store(self, trajectory, step, result, blackjack):

        if self.size >= self.max_batch:
            self.storage = self.storage[1:]
            self.result = self.result[1:]
            self.step = self.step[1:]
            self.blackjack = self.blackjack[1:]
            # self.double = self.double[1:]
            # self.surrender = self.surrender[1:]
            self.size -= 1

        self.storage.append(trajectory)
        self.result.append(result)
        self.step.append(step)
        self.blackjack.append(blackjack)
        # self.double.append(double)
        # self.surrender.append(surrender)
        self.size += 1

    def get_data(self, ind):

        return self.storage[ind]

    def get_step(self, ind):

        return self.step[ind]

    def get_performance(self):
        success_count = 0
        fail_count = 0
        tie_count = 0
        for i in range(self.size):
            if self.result[i] == 1:
                success_count += 1
            elif self.result[i] == -1:
                fail_count += 1
            elif self.result[i] == 0:
                tie_count += 1
        success_rate = success_count / self.size
        fail_rate = fail_count / self.size
        tie_rate = tie_count / self.size
        return success_rate, fail_rate, tie_rate, np.mean(self.step), np.mean(self.blackjack)  #, np.mean(self.double), np.mean(self.surrender)
