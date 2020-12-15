import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from tensorflow import keras
from ns3gym import ns3env
from tcp_base import TcpTimeBased
from tcp_newreno import TcpNewReno
from utility import *


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
simTime = 1000
stepTime = 0.5
seed = 12
simArgs = {"--duration": simTime, }
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim,
                    simSeed=seed, simArgs=simArgs, debug=debug)
env.reset()

state_space = env.observation_space.shape[0]
print(state_space)
state_trans_space = 3
action_space = 5


def get_agent(state):
    socketUuid = state[0]
    tcpEnvType = state[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    print(tcpAgent)
    if tcpAgent is None:
        if tcpEnvType == 0:
            # event-based = 0
            tcpAgent = TcpNewReno()
        else:
            tcpAgent = TcpTimeBased()
        tcpAgent.set_spaces(get_agent.state_space, get_agent.action_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent


U_new = 0
# hyper parameters
total_episodes = 10
max_env_steps = 10000
env._max_episode_steps = max_env_steps

epsilon = 0.01# exploration rate
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
action_mapping = {0: 0, 1: 536, 2: -536, 3: 268, 4: -268}
print_freq = 1

# loss
loss_window_size = 100
lt_1 = LossTrace(1, loss_window_size)

# initalize
get_agent.tcpAgents = {}
get_agent.state_space = state_trans_space
get_agent.action_space = action_space

model = keras.Sequential()
model.add(keras.layers.Dense(state_trans_space, input_shape=(state_trans_space,), activation='relu'))
model.add(keras.layers.Dense(state_trans_space * 3, input_shape=(state_trans_space,), activation='relu'))
model.add(keras.layers.Dense(state_trans_space * 3, input_shape=(state_trans_space * 3,), activation='relu'))
model.add(keras.layers.Dense(action_space, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights("./save_weights/_model.h5")

trace = Traces(1)
trace_2 = Traces(2)
default_cwnd = 536
total_reward = []

rtt_alpha = 0.6
cwnd_alpha = 0.6
rtt_ewma = EWMATrace(rtt_alpha)
cwnd_ewma = EWMATrace(cwnd_alpha)

state_trace = StateTrace(1)

for e in range(total_episodes):
    state = env.reset()
    cwnd = state[5]
    state_trans = extract_state(state, cwnd_ewma, rtt_ewma, lt_1)
    state = np.reshape(state, [1, state_space])
    state_trans = np.reshape(state_trans, [1, state_trans_space])
    reward_sum = 0
    seg_acked = 0
    lt_1.clear()
    U_old = 0
    No_step = 0

    for time in range(max_env_steps):
        lt_1.step()
        # choose action
        if np.random.rand(1) < epsilon:
            action_idx = np.random.randint(5)
        else:
            action_idx = np.argmax(model.predict(state_trans)[0])
        # print("selected action: ", action_mapping[action_idx])

        action = action_mapping[action_idx]
        new_cwnd = cwnd + action
        if new_cwnd < default_cwnd:
            new_cwnd = default_cwnd

        if trace.check_validate_np(state):
            trace.add_action(action)

        if trace_2.check_validate_np(state):
            trace_2.add_action(action)

        new_ssThresh = np.int(cwnd / 2)
        actions = [new_ssThresh, new_cwnd]
        U_reward, U_old = get_reward(state, lt_1, U_old)
        # step
        next_state, reward, done, info = env.step(actions)
        cwnd = next_state[5]
        state_trace.add_state(next_state)
        #print(next_state)
        #print(lt_1.get_error_prob())

        if next_state[11] == 0:
            lt_1.loss(next_state[0], No_step)

        if trace.check_validate(next_state):
            trace.add_cwnd(next_state[5])

        if trace_2.check_validate(next_state):
            trace_2.add_cwnd(next_state[5])

        if done:
            print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
                  .format(e, total_episodes, time, reward_sum, epsilon))
            total_reward.append(reward_sum)
            print(reward_sum / time)
            break
        next_state_trans = extract_state(next_state, cwnd_ewma, rtt_ewma, lt_1)
        next_state = np.reshape(next_state, [1, state_space])
        next_state_trans = np.reshape(next_state_trans, [1, state_trans_space])

        # Train
        target = U_reward
        if not done:
            target = (U_reward + 0.95 * np.amax(model.predict(next_state_trans)[0]))
        #target_f = model.predict(state_trans)
        #target_f[0][action_idx] = target

        #if state[0, 0] == 1:
        #    model.fit(state_trans, target_f, epochs=1, verbose=0)

        state = next_state
        if trace.check_validate_np(state):
            seg = state[0, 7]
            rtt = state[0, 9]
            seg_acked += seg
            reward_sum += U_reward
            trace.add_reward(U_reward)
            trace.add_rtt(rtt)
            # total_reward.append(reward_sum)
        if trace_2.check_validate_np(state):
            rtt = state[0, 9]
            trace_2.add_reward(U_reward)
            trace_2.add_rtt(rtt)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        No_step += 1
        if No_step % 100 == 0:
            print (No_step , state[0, 2])

    if e % print_freq == 0:
        print(seg_acked * default_cwnd)
        seg_acked = 0
        #trace.print_history()
        state_trace.print_history(e)
        state_trace.reset()
        # trace_2.print_history()

print(total_reward)
# save current model
