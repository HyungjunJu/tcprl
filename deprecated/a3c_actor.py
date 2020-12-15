import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Lambda
import tensorflow as tf


## Actor Neural Network
def build_network(state_dim, action_dim, action_bound):
    print(state_dim, action_dim , action_bound)
    state_input = Input((state_dim,))
    h1 = Dense(64, activation='relu')(state_input)
    h2 = Dense(32, activation='relu')(h1)
    h3 = Dense(16, activation='relu')(h2)
    output = Dense(3, activation='softmax')(h3)
    model = Model(state_input, output)
    # model.summary()
    model._make_predict_function()
    return model, model.trainable_weights, state_input


class Global_Actor(object):
    """
        Global Actor NN, Just need parameters, no need train
    """
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        #create global actor NN
        self.model, self.theta, self.states = build_network(self.state_dim, self.action_dim, self.action_bound)

    ## calculate average value in actor NN
    def predict(self, state):
        mu_a, _ = self.model.predict(np.reshape(state, [1, self.state_dim]))
        return mu_a[0]

    ## Save actor NN weights
    def save_weights(self, path):
        self.model.save_weights(path)

    ## Load actor NN weights
    def load_weights(self, path):
        self.model.load_weights(path+'_actor.h5')


class Worker_Actor(object):
    """
        Worker actor NN
    """
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, entropy_beta, global_actor):
        self.sess = sess
        self.global_actor = global_actor
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        # set standard deviation min and max value
        self.std_bound = [1e-2, 1]
        # create worker's actor nn
        self.model, self.theta, self.states = build_network(self.state_dim, self.action_dim, self.action_bound)

        # Placeholder for save policy and advantage
        self.actions = tf.placeholder(tf.float32, [None, self.action_dim])
        self.advantages = tf.placeholder(tf.float32, [None, 1])

        # policy pdf and entropy
        mu_a, std_a = self.model.output
        log_policy_pdf, entropy = self.log_pdf(mu_a, std_a, self.actions)

        # worker's loss function and gradients
        loss_policy = log_policy_pdf * self.advantages
        loss = tf.reduce_sum(-loss_policy - entropy_beta * entropy)
        dj_dtheta = tf.gradients(loss, self.theta)

        # gradient clipping
        dj_dtheta, _ = tf.clip_by_global_norm(dj_dtheta, 40)

        # update global NN by worker's gradient
        grads = zip(dj_dtheta, self.global_actor.theta)
        self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

    ## Log-policy pdf and entropy calcuation
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.log(var * 2 * np.pi)
        entropy = 0.5 * (tf.log(2 * np.pi * std ** 2) + 1.0)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True), tf.reduce_sum(entropy, 1, keepdims=True)

    ## select action from actor nn output by probability -- need to be changed
    def get_action(self, state, sess):
        print(state, self.state_dim)
        with sess.as_default():
            with sess.graph.as_default():
                input = np.reshape(state, [1, self.state_dim])
                mu_a, std_a = self.model.predict(input)
                mu_a = mu_a[0]
                std_a = std_a[0]
                std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
                action = np.random.normal(mu_a, std_a, size=self.action_dim)
                return action

    ## train worker actor NN
    def train(self, states, actions, advantages):
        self.sess.run(self.actor_optimizer, feed_dict={
            self.states: states,
            self.actions: actions,
            self.advantages: advantages
        })
