import sys
from ns3gym import ns3env
from utility import *
from tcprl_agent import *

EPISODES = 1000

# for ns3 environment
startSim = True
iterationNum = 1

port = 5555
simTime = 100000
stepTime = 0.5
seed = 12
simArgs = {"--duration": simTime, }
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim,
                    simSeed=seed, simArgs=simArgs, debug=debug)
print_freq = 1
weight_save_freq = 10

# for state Representation
rtt_alpha = 0.6
cwnd_alpha = 0.6

loss_window_size = 100
loss_trace = LossTrace(1, loss_window_size)
default_cwnd = 536

# action Mapping
action_mapping = {0: 0, 1: 536, 2: -536, 3: 268, 4: -268}

# default info
state_size = 3
action_size = len(action_mapping)

# Agent selection
# DeepSarsa = 1, A2C = 2
agent_type = 2
if agent_type == 1:
    agent = DeepSARSAAgent(state_size, action_size)
elif agent_type == 2:
    agent = A2CAgent(state_size, action_size)
else:
    agent = TCPAgent(state_size, action_size)

agent.load_weights()

if __name__ == "__main__":
    cwnd_max = env.action_space.high[0]
    trace = Traces(1)
    state_trace = StateTrace(1)
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        state = env.reset()
        cwnd = state[5]
        cwnd_ewma, rtt_ewma = EWMATrace(cwnd_alpha), EWMATrace(rtt_alpha)
        loss_trace.clear()
        U_old = 0
        No_step = 0
        seg_acked = 0
        score = 0

        state_trans = extract_state_without_ewma(state, cwnd_ewma, rtt_ewma, loss_trace)
        state = np.reshape(state, [1, len(state)])
        state_trans_normalized = normalize_state(state_trans, state, cwnd_max)
        state_trans_normalized = np.reshape(state_trans_normalized, [1, state_size])

        while not done:
            loss_trace.step()
            # choose action from agent
            action_idx = agent.get_action(state_trans_normalized)
            action_new = action_mapping[action_idx]
            new_cwnd = cwnd + action_new
            if new_cwnd < default_cwnd:
                new_cwnd = default_cwnd
            new_ssThresh = np.int(cwnd /2)
            action = [new_ssThresh, new_cwnd]
            next_state, reward, done, info = env.step(action)
            next_state_trans = extract_state(next_state, cwnd_ewma, rtt_ewma, loss_trace)
            next_state_reshaped = np.reshape(next_state, [1, len(next_state)])
            next_state_trans_normalized = normalize_state(next_state_trans, next_state_reshaped, cwnd_max)
            U_reward, U_old = get_reward_default(next_state, loss_trace, U_old)
            cwnd = next_state[5]
            state_trace.add_state(next_state)

            if next_state[11] == 0:
                loss_trace.loss(next_state[0], No_step)

            if trace.check_validate(next_state):
                trace.add_cwnd(next_state[5])

            next_state_trans_normalized = np.reshape(next_state_trans_normalized, [1, state_size])
            # agent.train_model(state_trans_normalized, action_idx, U_reward, next_state_trans_normalized, done)
            score += U_reward
            state = next_state
            state_trans = next_state_trans
            state_trans_normalized = next_state_trans_normalized

            if trace.check_validate(state):
                seg = state[7]
                rtt = state[9]
                seg_acked += seg
                trace.add_rtt(rtt)
                trace.add_cwnd(cwnd)

            No_step += 1
            if done:
                print("episode: {}/{}, time:{}, reward:{}".format(e, EPISODES, No_step, score))
                scores.append(score)
                episodes.append(e)

            agent.end_of_step()

        if e % print_freq == 0:
            print(seg_acked * default_cwnd)
            seg_acked = 0
            state_trace.print_history(e)
            state_trace.reset()

        if e % weight_save_freq == 0:
            agent.save_weights()

