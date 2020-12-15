import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from tensorflow import keras
from ns3gym import ns3env
from tcp_base import TcpTimeBased
from tcp_newreno import TcpNewReno

parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')

args = parser.parse_args()

startSim = True
iterationNum = 1

port = 5555
simTime = 10
stepTime = 0.5
seed = 12
simArgs = {"--duration": simTime, }
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
env.reset()

state_space = env.observation_space
action_space = 3


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
        plt.plot(range(len(cwnd_history)), cwnd_history, label='Cwnd', marker="", linestyle='-', color='red')
        plt.subplot(312)
        plt.plot(range(len(rtt_history)), rtt_history, label='Rtt', marker="", linestyle="-")  # , color='y')
        plt.subplot(313)
        plt.plot(range(len(reward_history)), reward_history, label='Reward', marker="", linestyle='-')

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


def get_agent(state):
    socketUuid = state[0]
    tcpEnvType = state[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        if tcpEnvType == 0:
            # event-based = 0
            tcpAgent = TcpNewReno()
        else:
            tcpAgent = TcpTimeBased()
        tcpAgent.set_spaces(get_agent.state_space, get_agent.action_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent


delta_tp = 1
delta_rtt = 1
delta_loss = 1
avg_tp_default = 1


def tp_calc(cwnd, rtt):
    avg_tp = 3/4 * cwnd / rtt
    return avg_tp


def utility_function(s):
    cwnd = s[0, 5]
    rtt = s[0, 9]
    avg_tp = tp_calc(cwnd, rtt)
    if avg_tp == 0:
        avg_tp = avg_tp_default
    utility = delta_tp * np.log(avg_tp) - delta_rtt * np.log(rtt) # - delta_loss * np.log(rtt)
    return utility


def get_reward(s):
    u = utility_function(s)
    return u


# hyper parameters
total_episodes = 100
max_env_steps = 3000
env._max_episode_steps = max_env_steps

epsilon = 1.0 # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999

learning_rate = 0.001

time_history = []

reward_history = []
cwnd_history = []
action_history = []
rtt_history = []

No_step = 0
reward_sum = 0
done = False
info = None
action_mapping = {0: 0, 1: 600, 2: -60}
print_freq = 1

# initalize
get_agent.tcpAgents = {}
get_agent.state_space = state_space
get_agent.action_space = action_space

model = keras.Sequential()
model.add(keras.layers.Dense(state_space, input_shape=(state_space,), activation='relu'))
model.add(keras.layers.Dense(state_space, input_shape=(state_space,), activation='relu'))
model.add(keras.layers.Dense(action_space, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

trace = Traces(0)

for e in range(total_episodes):
    state = env.reset()
    cwnd = state[5]
    state = np.reshape(state, [1, state_space])
    reward_sum = 0

    for time in range(max_env_steps):
        # choose action
        if np.random.rand(1) < epsilon:
            action_idx = np.random.randint(3)
        else:
            action_idx = np.argmax(model.predict(state)[0])
        print("selected action: ", action_mapping[action_idx])

        action = action_mapping[action_idx]
        new_cwnd = cwnd + action
        if trace.check_validate_np(state):
            trace.add_action(action)

        new_ssThresh = np.int(cwnd / 2)
        actions = [new_ssThresh, new_cwnd]
        U_reward = get_reward(state)
        # step
        next_state, reward, done, info = env.step(actions)

        if trace.check_validate(next_state):
            trace.add_cwnd(next_state[5])

        if done:
            print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
                  .format(e, total_episodes, time, reward_sum, epsilon))
            break

        next_state = np.reshape(next_state, [1, state_space])

        # Train
        target = U_reward
        if not done:
            target = (U_reward + 0.95 * np.amax(model.predict(next_state)[0]))
        target_f = model.predict(state)
        target_f[0][action_idx] = target

        if state[0, 0] == 1:
            model.fit(state, target_f, epochs=1, verbose=0)

        state = next_state
        if trace.check_validate_np(state):
            seg = state[0, 5]
            rtt = state[0, 9]
            reward_sum += U_reward
            trace.add_reward(U_reward)
            trace.add_rtt(rtt)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        No_step += 1

    if e % print_freq == 0:
        trace.print_history()


