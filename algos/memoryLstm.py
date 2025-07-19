from collections import namedtuple
import random
import torch

ExperienceLstm = namedtuple('ExperienceLstm',
                            ('states', 'actions', 'next_states', 'rewards', "past_inf"))

agent_observation = namedtuple("agent_ob", ["pod", "res_time"])


class ReplayMemoryLstm:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = ExperienceLstm(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    replay = ReplayMemoryLstm(2000)
    replay.push(1, 2, 3, 4, 5)
    torch.save(replay, "111.pt")
