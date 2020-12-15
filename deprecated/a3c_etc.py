import matplotlib as mpl
import matplotlib.pyplot as plt


class EWMATrace:
    def __init__(self, alpha):
        self.alpha = alpha
        self.current = 0

    def get_current(self, value):
        if self.current == 0: return value
        return self.current

    def next(self, value):
        self.current = self.alpha * value + self.current * (1 - self.alpha)
        return self.current

    def clear(self):
        self.current = 0


class LossItem:
    def __init__(self, step_no, lifetime):
        self.step_no = step_no
        self.lifetime = lifetime

    def discount(self):
        self.lifetime -= 1

    def check_lifetime(self):
        return self.lifetime == 0


class LossTrace:
    def __init__(self, socketUuid, window_size):
        self.socketUuid = socketUuid
        self.window_size = window_size
        self.list = []

    def loss(self, socketUuid, step_no):
        if self.socketUuid != socketUuid:
            return
        self.list.append(LossItem(step_no, self.window_size))

    def step(self):
        removed = []
        for loss in self.list:
            loss.discount()
            if loss.check_lifetime():
                removed.append(loss)
        for loss in removed:
            self.list.remove(loss)

    def get_error_prob(self):
        sum = 0
        for lossItem in self.list:
            sum += lossItem.lifetime
        return sum / (self.window_size ** 2)

    def clear(self):
        self.list.clear()


class Traces:
    def __init__(self, socketUuid):
        self.socketUuid = socketUuid
        self.time_history = []
        self.reward_history = []
        self.cwnd_history = []
        self.action_history = []
        self.rtt_history = []

    def add_reward(self, reward):
        self.reward_history.append(reward)

    def add_cwnd(self, cwnd):
        self.cwnd_history.append(cwnd)

    def add_action(self, action):
        self.action_history.append(action)

    def add_rtt(self, rtt):
        self.rtt_history.append(rtt)

    def check_validate_np(self, state):
        return state[0, 0] == self.socketUuid

    def check_validate(self, state):
        return state[0] == self.socketUuid

    def print_history(self):
        mpl.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.grid(True, linestyle='--')
        plt.title('Learning Performance')
        plt.subplot(311)
        plt.plot(range(len(self.cwnd_history)), self.cwnd_history, label='Cwnd', marker="", linestyle='-', color='red')
        plt.subplot(312)
        plt.plot(range(len(self.rtt_history)), self.rtt_history, label='Rtt', marker="", linestyle="-")  # , color='y')
        plt.subplot(313)
        plt.plot(range(len(self.reward_history)), self.reward_history, label='Reward', marker="", linestyle='-')

        plt.xlabel('Episode')
        plt.ylabel('value')
        plt.legend(prop={'size': 12})
        plt.savefig('learning.pdf', bbox_inches='tight')
        plt.show()
        self.clear_history()

    def clear_history(self):
        self.reward_history.clear()
        self.cwnd_history.clear()
        self.action_history.clear()
        self.rtt_history.clear()

