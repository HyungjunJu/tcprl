import sys
from ns3gym import ns3env
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from utility import *

EPISODES = 100


class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        self.discount_factor = 0.999
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()
        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear', kernel_initializer='he_uniform'))
        critic.summary()
        return critic

    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        print(state)
        #print(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [],
                           updates=updates)
        return train

    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])
        loss = K.mean(K.square(target - self.critic.output))
        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)
        return train

    def train_model(self, state, action, reward, next_state, done):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        self.actor_updater([state, act, advantage])
        self.critic_updater([state, target])


startSim = True
iterationNum = 1

port = 5555
simTime = 1000
stepTime = 0.5
seed = 12
simArgs = {"--duration": simTime, }
debug = False

action_mapping = {0: 0, 1: 536, 2: -536, 3: 268, 4: -268}

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim,
                    simSeed=seed, simArgs=simArgs, debug=debug)

rtt_alpha = 0.6
cwnd_alpha = 0.6

loss_window_size = 100
lt_1 = LossTrace(1, loss_window_size)
default_cwnd = 536

trace = Traces(1)
state_trace = StateTrace(1)
print_feq = 1

epsilon = 0.99
epsilon_min = 0.01
epsilon_decay = 0.999

if __name__ == "__main__":
    state_size = 3
    action_size = 5
    seg_acked = 0

    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []
    cwnd_ewma, rtt_ewma = EWMATrace(cwnd_alpha), EWMATrace(rtt_alpha)
    lt_1.clear()

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        cwnd = state[5]
        cwnd_max = env.action_space.high[0]
        state_tr = extract_state_without_ewma(state, cwnd_ewma, rtt_ewma, lt_1)
        state = np.reshape(state, [1, len(state)])
        state_tr = normalize_state(state_tr, state, cwnd_max)
        state_tr = np.reshape(state_tr, [1, state_size])
        U_old = 0
        No_step = 0

        while not done:
            lt_1.step()
            #if np.random.rand(1) < epsilon:
            #    action_temp = np.random.randint(5)
            #else:
            #    action_temp = agent.get_action(state_tr)
            action_temp = agent.get_action(state_tr)
            action_new = action_mapping[action_temp]
            new_cwnd = cwnd + action_new
            if new_cwnd < default_cwnd:
                new_cwnd = default_cwnd
            new_ssThresh = np.int(cwnd / 2)
            action = [new_ssThresh, new_cwnd]
            next_state, reward, done, info = env.step(action)
            next_state_tr = extract_state(next_state, cwnd_ewma, rtt_ewma, lt_1)
            reshaped_next_state = np.reshape(next_state, [1, len(next_state)])
            next_state_tr = normalize_state(next_state_tr, reshaped_next_state, cwnd_max)
            U_reward, U_old = get_reward_default(next_state, lt_1, U_old)
            cwnd = next_state[5]
            state_trace.add_state(next_state)

            if next_state[11] == 0:
                lt_1.loss(next_state[0], No_step)

            if trace.check_validate(next_state):
                trace.add_cwnd(next_state[5])

            next_state_tr = np.reshape(next_state_tr, [1, state_size])
            agent.train_model(state_tr, action_temp, U_reward, next_state_tr, done)
            score += reward
            state = next_state
            state_tr = next_state_tr

            if trace.check_validate(state):
                seg = state[7]
                rtt = state[9]
                cwnd = state[5]
                seg_acked += seg
                trace.add_rtt(rtt)
                trace.add_cwnd(cwnd)

            No_step += 1
            if done:
                print("episode: {}/{}, time:{}, reward:{}".format(e, EPISODES, No_step, score))
                scores.append(score)
                episodes.append(e)

        if e % print_feq == 0:
            print(seg_acked * default_cwnd)
            seg_acked = 0
            state_trace.print_history(e)
            state_trace.reset()


