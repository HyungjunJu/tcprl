import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

delta_tp = 1
delta_rtt = 1
delta_loss = 50
avg_tp_default = 1


class EWMATrace:
    def __init__(self, alpha):
        self.alpha = alpha
        self.current = 0

    def next(self, value):
        self.current = self.alpha * value + self.current * (1 - self.alpha)
        return self.current

    def get_current(self):
        return self.current

    def clear(self):
        self.current = 0


class LossTrace:
    def __init__(self, socketUuid, window_size):
        self.socketUuid = socketUuid
        self.window_size = window_size
        self.list = []

    def loss(self, socketUuid, step_no):
        if self.socketUuid != socketUuid:
            return
        self.list.append(Loss(step_no, self.window_size))

    def step(self):
        removed = []
        for loss in self.list:
            loss.discount()
            if loss.check_lifetime():
                removed.append(loss)
        for loss in removed:
            self.list.remove(loss)

    def get_error_prob(self):
        # return len(self.list) / self.window_size
        sum = 0
        for lossItem in self.list:
            sum = sum + lossItem.lifetime
        return sum / (self.window_size ** 2)

    def clear(self):
        self.list.clear()


class Loss:
    def __init__(self, step_no, lifetime):
        self.step_no = step_no
        self.lifetime = lifetime

    def discount(self):
        self.lifetime -= 1

    def check_lifetime(self):
        return self.lifetime == 0


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


class StateTrace:
    def __init__(self, socketUuid):
        self.socketUuid = socketUuid
        self.stateHistory = []
        self.time_max = -1

    def add_state(self, state):
        if self.socketUuid == state[0]:
            self.stateHistory.append(state)

    def reset(self):
        self.stateHistory.clear()

    def extract(self, idx):
        output = []
        for state in self.stateHistory:
            output.append(state[idx])
        return output

    def print_history(self, idx):
        # print 2(x axis), 5, 9, 10(once), 11(loss - 0)
        try:
            f = open(f"result/result{idx}.txt", 'w')
        except:
            return
        str_time = "time(ms)"
        str_cwnd = "cwnd(bytes)"
        str_rtt = "rtt(ms)"
        str_loss = "loss(0-loss)"
        f.write(f"idx\t{str_time:10s}\t{str_cwnd:10s}\t{str_rtt:10s}\t{str_loss:10s}\n")
        self.stateHistory.pop(len(self.stateHistory) - 1)
        i = 0
        for item in self.stateHistory:
            data = f"{i}\t{item[2] / 1000:10.2f}\t{item[5]:10d}\t{item[9] / 1000:10.2f}\t{item[11]}\n"
            i += 1
            f.write(data)
        f.close()
        return


def tp_calc(cwnd, rtt):
    avg_tp = 3 / 4 * cwnd / rtt
    return avg_tp


def utility_function_default(s, losstrace):
    cwnd = s[5]
    rtt = s[9]
    # avg_tp = tp_calc(cwnd, rtt)
    avg_tp = cwnd
    if avg_tp == 0:
        avg_tp = avg_tp_default
    utility = delta_tp * np.log(avg_tp) \
              - delta_rtt * np.log(rtt) \
              + delta_loss * np.log(1 - losstrace.get_error_prob())
    return utility


def utility_function(s, losstrace):
    cwnd = s[0, 5]
    rtt = s[0, 9]
    # avg_tp = tp_calc(cwnd, rtt)
    avg_tp = cwnd
    if avg_tp == 0:
        avg_tp = avg_tp_default
    utility = delta_tp * np.log(avg_tp) \
              - delta_rtt * np.log(rtt) \
              + delta_loss * np.log(1 - losstrace.get_error_prob())
    return utility


def get_reward_default(s, losstrace, U_old):
    u = utility_function_default(s, losstrace)
    u_diff = u - U_old
    # print(u_diff)
    if u_diff >= 0:
        # return 1, u
        return u, u
    else:
        # return -1, u
        return u, u
    # return u


def get_reward(s, losstrace, U_old):
    u = utility_function(s, losstrace)
    u_diff = u - U_old
    print(u_diff)
    if u_diff > 0:
        return 1, u
    else:
        return -1, u
    # return u, u


def extract_state_without_ewma(state, cwnd_ewma, rtt_ewma, losstrace):
    rtt_e = rtt_ewma.get_current()
    cwnd_e = cwnd_ewma.get_current()
    loss = losstrace.get_error_prob()
    output = [rtt_e, cwnd_e, loss]
    return output


def extract_state(state, cwnd_ewma, rtt_ewma, losstrace):
    rtt = state[9]
    rtt_min = state[10]
    rtt_diff = rtt - rtt_min
    # rtt_diff /= 1000
    rtt_e = rtt_ewma.next(rtt_diff)
    cwnd = state[5]
    cwnd_e = cwnd_ewma.next(cwnd)
    loss = losstrace.get_error_prob()
    output = [rtt_e, cwnd_e, loss]
    return output


_rtt_max = 1000
_cwnd_max = 100000


def normalize_state(state, default_state, max_cwnd):
    output = [0, 0, 0]
    output[0] = state[0] / default_state[0, 10]
    output[1] = state[1] / max_cwnd
    output[2] = state[2]
    return output
