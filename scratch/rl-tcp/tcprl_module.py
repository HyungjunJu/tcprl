from tcprl_agent import *
from utility import *
import threading
import os
import time
import multiprocessing as mp
from functools import partial


global episode
episode = 0
EPISODES = 1000

startSim = True
iterationNum = 1

port = 5555
simTime = 1000000
stepTime = 0.5
seed = 12
simArgs = {"--duration": simTime, }
debug = False

print_result = True
weight_save_frequency = 10

# for state Representation
rtt_alpha = 0.6
cwnd_alpha = 0.6

loss_window_size = 100
default_cwnd = 536

# action Mapping
action_mapping = {0: 0, 1: 536, 2: -536, 3: 536 * 0.5, 4: -536 * 0.5}

# default info
state_size = 3
action_size = len(action_mapping)


class A3CWorker:
#class A3CWorker(threading.Thread):
    def __init__(self, agent, sess, global_agent, state_size, action_size, threadidx, seed_adder, graph, isTrain=True):
        ##threading.Thread.__init__(self)
        self.sess = sess
        self.agent = agent
        #self.global_agent = global_agent
        self.state_size = state_size
        self.action_size = action_size
        self.local_actor, self.local_critic = agent.actor, agent.critic
        self.port_adder = threadidx
        self.seed_adder = seed_adder
        self.t_max = 10
        self.t = 0
        self.isTrain = isTrain
        self.batch_states, self.batch_rewards, self.batch_actions = [], [], []
        #
        # a, c = global_agent.get_weights()
        # self.agent.set_weights(a, c)
        self.env = ns3env.Ns3Env(port=port + threadidx, stepTime=stepTime, startSim=startSim,
                            simSeed=seed, simArgs=simArgs, debug=debug)
        print("environment initialized", self.port_adder)

    def save_batch(self, state, action, reward):
        self.batch_states.append(state)
        self.batch_actions.append(action)
        self.batch_rewards.append(reward)

    def extract_batch(self):
        return self.batch_states, self.batch_actions, self.batch_rewards

    def clear_batch(self):
        self.batch_states.clear()
        self.batch_actions.clear()
        self.batch_rewards.clear()

    def run(self, lock, global_agent):
        episode = 0
        cwnd_max = self.env.action_space.high[0]
        trace = Traces(1)
        state_trace = StateTrace(1)
        scores, episodes = [], []
        lock.acquire()
        a, c = global_agent.get_weights()
        self.agent.set_weights(a, c)
        lock.release()

        try:
            while episode < EPISODES:
                print("start episode", self.port_adder, episode)
                done = False
                No_step, U_old, seg_acked, score = 0, 0, 0, 0
                state = self.env.reset()
                cwnd_ewma, rtt_ewma = EWMATrace(cwnd_alpha), EWMATrace(rtt_alpha)
                loss_trace = LossTrace(1, loss_window_size)
                cwnd = state[5]

                state_trans = extract_state_without_ewma(state, cwnd_ewma, rtt_ewma, loss_trace)
                state = np.reshape(state, [1, len(state)])
                state_trans_normalized = normalize_state(state_trans, state, cwnd_max)
                state_trans_normalized = np.reshape(state_trans_normalized, [1, state_size])
                lock.acquire()
                a, c = global_agent.get_weights()
                lock.release()
                self.agent.set_weights(a, c)

                while not done:
                    print("step", self.port_adder)
                    loss_trace.step()
                    action_idx = self.agent.get_action(state_trans_normalized)
                    action_new = np.int(action_mapping[action_idx])
                    new_cwnd = cwnd + action_new
                    if new_cwnd < default_cwnd:
                        new_cwnd = default_cwnd
                    new_ssThresh = np.int(cwnd / 2)
                    action = [new_ssThresh, new_cwnd]
                    next_state, reward, done, info = self.env.step(action)
                    next_state_trans = extract_state(next_state, cwnd_ewma, rtt_ewma, loss_trace)
                    next_state_reshaped = np.reshape(next_state, [1, len(next_state)])
                    next_state_trans_normalized = normalize_state(next_state_trans, next_state_reshaped, cwnd_max)
                    U_reward, U_old = get_reward_default(next_state, loss_trace, U_old)
                    cwnd = next_state[5]
                    state_trace.add_state(next_state)
                    if next_state[11] == 0:
                        loss_trace.loss(next_state[0], No_step)

                    next_state_trans_normalized = np.reshape(next_state_trans_normalized, [1, state_size])
                    action_reshaped = np.zeros(self.action_size)
                    action_reshaped[action_idx] = 1
                    self.save_batch(state_trans_normalized,
                                    np.reshape(action_reshaped, [1, self.action_size]),
                                    U_reward)
                    score += U_reward
                    state = next_state
                    state_trans = next_state_trans
                    state_trans_normalized = next_state_trans_normalized
                    self.t += 1
                    if self.t == self.t_max:
                        # train
                        self.t = 0
                        states, actions, rewards = self.extract_batch()
                        self.agent.train_model(states, actions, rewards, next_state_trans_normalized, done)
                        a, c = self.agent.get_weights()
                        lock.acquire()
                        global_agent.set_weights(a, c)
                        lock.release()
                        self.clear_batch()

                    if trace.check_validate(state):
                        seg = state[7]
                        rtt = state[9]
                        seg_acked += seg
                        trace.add_rtt(rtt)
                        trace.add_cwnd(cwnd)

                    No_step += 1
                    if done:
                        print("episode: {}/{}, time:{}, reward:{}".format(episode, EPISODES, No_step, score))
                        scores.append(score)
                        episodes.append(No_step)
                        episode += 1

                    self.agent.end_of_step()

                if print_result:
                    print(seg_acked * default_cwnd)
                    if self.port_adder == 0:
                        state_trace.print_history(episode)

                state_trace.reset()
                seg_acked = 0
                if episode % weight_save_frequency == 0:
                    lock.acquire()
                    global_agent.save_weights()
                    lock.release()
        except Exception as e:
            print("Error", e)
        print("thread out", self.port_adder)


def A3C_execute(isTrain=True):
    sess = tf.Session()
    K.set_session(sess)
    thread_count = os.cpu_count()
    graph = tf.get_default_graph()
    workers = []
    t_max = 10
    global_agent = A3CAgent(state_size, action_size, graph, t_max)
    lock = mp.Lock()
    if isTrain:
        # for i in range(thread_count):
        #     worker = A3CWorker(A3CAgent(state_size, action_size, graph, t_max),
        #                        sess, global_agent, state_size, action_size, i, i, isTrain)
        #     workers.append(worker)
        # for w in workers:
        #     time.sleep(1)
        #     w.start()
        # for w in workers:
        #     w.join()
        pools = []
        for i in range(thread_count):
            worker = A3CWorker(A3CAgent(state_size, action_size, graph, t_max),
                                sess, global_agent, state_size, action_size, i, i, isTrain)
            workers.append(worker)
            p = mp.Process(target=worker.run, args=(lock, global_agent))
            p.start()
            pools.append(p)
        return pools


def normal_execute(agent_type=1, isTrain=True, port_adder=0, seed_adder=0):
    if agent_type == 1:
        agent = DeepSARSAAgent(state_size, action_size)
    elif agent_type == 2:
        agent = A2CAgent(state_size, action_size)
    elif agent_type == 3:
        return A3C_execute(isTrain)
    else:
        agent = TCPAgent(state_size, action_size)

    env = ns3env.Ns3Env(port=port + port_adder, stepTime=stepTime, startSim=startSim,
                        simSeed=seed + seed_adder, simArgs=simArgs, debug=debug)

    cwnd_max = env.action_space.high[0]
    trace = Traces(1)
    state_trace = StateTrace(1)
    policy_trace = PolicyTrace()
    scores, episodes = [], []
    loss_trace = LossTrace(1, loss_window_size)

    if not isTrain:
        agent.load_weights()

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
            action_idx = agent.get_action(state_trans_normalized)
            action_new = action_mapping[action_idx]
            new_cwnd = np.int(cwnd + action_new)
            if new_cwnd < default_cwnd:
                new_cwnd = default_cwnd
            new_ssThresh = np.int(cwnd / 2)
            action = [new_ssThresh, new_cwnd]
            next_state, reward, done, info = env.step(action)
            next_state_trans = extract_state(next_state, cwnd_ewma, rtt_ewma, loss_trace)
            next_state_reshaped = np.reshape(next_state, [1, len(next_state)])
            next_state_trans_normalized = normalize_state(next_state_trans, next_state_reshaped, cwnd_max)
            U_reward, U_old = get_reward_default(next_state, loss_trace, U_old)
            cwnd = next_state[5]
            state_trace.add_state(next_state)
            policy_trace.add_policy(agent.get_policy(state_trans_normalized))
            if next_state[11] == 0:
                loss_trace.loss(next_state[0], No_step)

            next_state_trans_normalized = np.reshape(next_state_trans_normalized, [1, state_size])
            if isTrain:
                agent.train_model(state_trans_normalized, action_idx, U_reward, next_state_trans_normalized, done)
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

        if print_result:
            print(seg_acked * default_cwnd)
            state_trace.print_history(e)
            policy_trace.print_policy(e)

        state_trace.reset()
        policy_trace.reset()
        seg_acked = 0
        if e % weight_save_frequency == 0:
            agent.save_weights()

