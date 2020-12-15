import sys
from ns3gym import ns3env
import pylab
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from utility import *


class TCPAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def get_action(self, state):
        return 0

    def train_model(self, state, action, reward, next_state, done):
        return

    def end_of_step(self):
        return

    def save_weights(self):
        return

    def load_weights(self):
        return


class DeepSARSAAgent(TCPAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.model = keras.Sequential()
        self.learning_rate = 0.001
        self.model.add(keras.layers.Dense(state_size, input_shape=(state_size,), activation='relu'))
        self.model.add(keras.layers.Dense(state_size * 3, input_shape=(state_size,), activation='relu'))
        self.model.add(keras.layers.Dense(state_size * 3, input_shape=(state_size * 3,), activation='relu'))
        self.model.add(keras.layers.Dense(action_size, activation='softmax'))
        self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.epsilon = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

    def get_action(self, state):
        if np.random.rand(1) < self.epsilon:
            action_idx = np.random.randint(5)
        else:
            action_idx = np.argmax(self.model.predict(state)[0])
        return action_idx

    def train_model(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = (reward + 0.95 * np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        if state[0, 0] == 1:
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def end_of_step(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_weights(self):
        self.model.save_weights("./save_weights/deepsarsa_model.h5")

    def load_weights(self):
        self.model.load_weights("./save_weights/deepsarsa_model.h5")


class A2CAgent(TCPAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
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
        # print(state)
        # print(policy)
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

        #print(state, act, advantage)
        self.actor_updater([state, act, advantage])
        #print(state, target)
        self.critic_updater([state, target])

    def save_weights(self):
        self.actor.save_weights("./save_weights/a2c_actor_model.h5")
        self.critic.save_weights("./save_weights/a2c_critic_model.h5")

    def load_weights(self):
        self.actor.load_weights("./save_weights/a2c_actor_model.h5")
        self.critic.load_weights("./save_weights/a2c_critic_model.h5")


class A3CAgent(TCPAgent):
    def __init__(self, state_size, action_size, graph, t_max):
        super().__init__(state_size, action_size)
        self.value_size = 1

        self.beta = 0.01 # for entropy bonus
        self.discount_factor = 0.999
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.graph = graph
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor._make_predict_function()
        self.critic._make_predict_function()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()
        self.t_max = t_max

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        #actor.summary()
        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear', kernel_initializer='he_uniform'))
        #critic.summary()
        return critic

    def get_action(self, state):
        with self.graph.as_default():
            policy = self.actor.predict(state, batch_size=1).flatten()
        # print(state)
        # print(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        policy = self.actor.output
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        cross_entropy = -K.sum(cross_entropy)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        #loss = -K.sum(cross_entropy)
        loss = cross_entropy + self.beta * entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [loss],
                           updates=updates)
        return train

    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])
        value = self.critic.output
        loss = K.mean(K.square(target - value))
        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [loss], updates=updates)
        return train

    def n_step_td_targets(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value
        for k in reversed(range(0, len(rewards))):
            cumulative = self.discount_factor * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def v_values(self, states):
        output = np.zeros([1, len(states)])
        for i in range(0, len(states)):
            output[0, i] = self.critic.predict(states[i])
        #print("output", output)
        return output

    def train_model(self, state, action, reward, next_state, done):
        with self.graph.as_default():
            next_v_value = self.critic.predict(next_state)[0]
            n_step_td_targets = self.n_step_td_targets(reward, next_v_value, done)
            v_values = self.v_values(state)
            advantages = n_step_td_targets - v_values

            for i in range(self.t_max):
                #print("idx", i, "state", state[i], "act", action[i], "adv", advantages[0, i])
                adv = [advantages[0, i]]
                self.actor_updater([state[i], action[i], adv])
                ntarget = [n_step_td_targets[i]]
                self.critic_updater([state[i], ntarget])

    def save_weights(self):
        self.actor.save_weights("./save_weights/a3c_actor_model.h5")
        self.critic.save_weights("./save_weights/a3c_critic_model.h5")

    def load_weights(self):
        self.actor.load_weights("./save_weights/a3c_actor_model.h5")
        self.critic.load_weights("./save_weights/a3c_critic_model.h5")

    def get_weights(self):
        return self.actor.get_weights(), self.critic.get_weights()

    def set_weights(self, actor, critic):
        print("set_weights")
        with self.graph.as_default():
            self.actor.set_weights(actor)
            self.critic.set_weights(critic)

