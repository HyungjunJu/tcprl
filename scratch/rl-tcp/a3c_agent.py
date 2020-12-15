import numpy as np
import tensorflow as tf
import keras.backend as K
import threading
import multiprocessing
import argparse

from ns3gym import ns3env
from a3c_actor import Global_Actor, Worker_Actor
from a3c_critic import Global_Critic, Worker_Critic
#from a3c_etc import *
from utility import *

# common variables for used in workers
global_episode_count = 0        # total episode counts
global_step = 0                 # total step counts
global_episode_reward = []      # save rewards

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
startSim = bool(args.start)
iterationNum = int(args.iterations)

port = 5555
simTime = 10  # seconds
stepTime = 0.5  # seconds
seed = 12
simArgs = {"--duration": simTime, }
debug = False

state_dimension = 3 #rtt_ewma, cwnd_ewma, lossrate
action_dimension = 5 #change cwnd

action_mapping = {0: 0, 1: 536, 2: -536, 3: 268, 4: -268}
action_max_bound = 1000


class A3Cagent(object):
    """
        Create Global NN
    """
    def __init__(self, env_name):
        # set tensorflow session
        self.sess = tf.Session()
        K.set_session(self.sess)

        # generate learning environment
        self.env_name = env_name
        self.WORKERS_NUM = multiprocessing.cpu_count()
        #env = gym.make(self.env_name)       # change here to ns3gym
        env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
        # state dimension
        #state_dim = env.observation_space.shape[0]
        state_dim = state_dimension
        # action dimension
        #action_dim = env.action_space.shape[0]
        action_dim = action_dimension
        # maximum action boundary
        #action_bound = env.action_space.high[0]
        action_bound = action_max_bound
        # generate global actor and critic nn
        self.global_actor = Global_Actor(state_dim, action_dim, action_bound)
        self.global_critic = Global_Critic(state_dim)

    def train(self, max_episode_num):
        workers = []
        # A3CWorker thread generate and append to list
        for i in range(self.WORKERS_NUM):
            worker_name = 'worker%i' % i
            workers.append(
                A3CWorker(worker_name, i+1,
                          self.sess, self.global_actor,
                          self.global_critic, max_episode_num))

        # session initalization for gradients
        self.sess.run(tf.global_variables_initializer())

        # look up list and start worker thread
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        # after learning, save global reward
        np.savetxt('./save_weights/_epi_reward.txt', global_episode_reward)
        print(global_episode_reward)

    def plot_result(self):
        plt.plot(global_episode_reward)
        plt.show()


class A3CWorker(threading.Thread):
    """
        create woreker thread
    """
    def __init__(self, worker_name, env_count, sess,
                 global_actor, global_critic, max_episode_num):
        threading.Thread.__init__(self)

        # hyperparameters
        self.GAMMA = 0.95
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.ENTROPY_BETA = 0.01
        self.t_MAX = 4 # n-step TD

        self.max_episode_num = max_episode_num

        #self.env = gym.make(env_name)
        self.env = ns3env.Ns3Env(port=(port+env_count), stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
        self.worker_name = worker_name
        self.sess = sess

        self.global_actor = global_actor
        self.global_critic = global_critic

        # state variable dimension
        #self.state_dim = self.env.observation_space.shape[0]
        self.state_dim = state_dimension
        # action dimension
        #self.action_dim = self.env.action_space.shape[0]
        self.action_dim = action_dimension
        # action maximum boundary
        #self.action_bound = int(self.env.action_space.high[0])
        self.action_bound = action_max_bound
        # create worker actor and critic NN
        self.worker_actor = Worker_Actor(self.sess, self.state_dim, self.action_dim,
                                         self.action_bound, self.ACTOR_LEARNING_RATE,
                                         self.ENTROPY_BETA, self.global_actor)
        self.worker_critic = Worker_Critic(self.sess, self.state_dim, self.action_dim,
                                           self.CRITIC_LEARNING_RATE, self.global_critic)
        # Copy Hyperparameters from Global NN to Worker NN
        self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
        self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

    ## calculage n-step td target
    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.GAMMA * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    ## extract data from batch
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx+1], axis=0)
        return unpack

    ## worker train function, run - thread
    def run(self):
        # declaration for common global variables
        global global_episode_count, global_step
        global global_episode_reward
        # print worker execution
        print(self.worker_name, "starts ----")
        trace = Traces(1)
        default_cwnd = 536
        rtt_alpha = 0.6
        cwnd_alpha = 0.6
        rtt_ewma = EWMATrace(rtt_alpha)
        cwnd_ewma = EWMATrace(cwnd_alpha)
        loss_window_size = 100
        losstrace = LossTrace(1, loss_window_size)
        seg_acked = 0

        # repeat episodes
        while global_episode_count <= int(self.max_episode_num):
            # initialize batch
            batch_state, batch_action, batch_reward = [], [], []
            # initialize episode
            step, episode_reward, done = 0, 0, False
            # reset environment and observe initial state
            state = self.env.reset()
            cwnd = state[5]
            U_old = 0
            losstrace.clear()
            # repeat episode
            while not done:
                # rendering
                # self.env.render()
                # extract action
                losstrace.step() # maybe socketUuid need
                state_tr = extract_state_without_ewma(state, cwnd_ewma, rtt_ewma, losstrace)
                action = self.worker_actor.get_action(state_tr, self.sess)
                print(":", action, ":")
                # action boundary clipping
                #action = int(np.clip(action, -self.action_bound, self.action_bound))
                new_cwnd = cwnd + action_mapping[action]
                if new_cwnd < default_cwnd:
                    new_cwnd = default_cwnd
                if trace.check_validate(state):
                    trace.add_action(action)
                new_ssThresh = np.int(cwnd / 2)
                actions = [new_ssThresh, new_cwnd]
                U_reward, U_old = get_reward(state, losstrace, U_old)

                # observe next state and reward
                next_state, reward, done, _ = self.env.step(actions)
                next_state_tr = extract_state(next_state, cwnd_ewma, rtt_ewma, losstrace)
                #reward_new = get_reward(state)
                cwnd = next_state[5]
                if next_state[11] == 0 & next_state[0] == 1:
                    losstrace.loss(1, step)
                if trace.check_validate(next_state):
                    trace.add_cwnd(cwnd)
                    trace.add_rtt(next_state[9])
                    trace.add_reward(U_reward)
                    seg_acked += next_state[7]

                # shape translation
                state = np.reshape(state_tr, [1, self.state_dim])
                reward = np.reshape(U_reward, [1, 1])
                action = np.reshape(action, [1, self.action_dim])
                # save batch
                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(reward)
                # update state
                state = next_state
                step += 1
                episode_reward += reward[0]

                # if batch is filled, start worker training
                with self.sess.as_default():
                    with self.sess.graph.as_default():  
                        if len(batch_state) == self.t_MAX or done:
                            # extract data from batch
                            states = self.unpack_batch(batch_state)
                            actions = self.unpack_batch(batch_action)
                            rewards = self.unpack_batch(batch_reward)
                            # clear batch
                            batch_state, batch_action, batch_reward = [], [], []
                           # calculate n-step TD target and advantages
                            next_state = np.reshape(next_state_tr, [1, self.state_dim])
                            next_v_value = self.worker_critic.model.predict(next_state)
                            n_step_td_targets = self.n_step_td_target(rewards, next_v_value, done)
                            v_values = self.worker_critic.model.predict(states)
                            advantages = n_step_td_targets - v_values
                            # update global critic and actor nn
                            self.worker_critic.train(states, n_step_td_targets)
                            self.worker_actor.train(states, actions, advantages)
                            # copy global parameter to worker nn
                            self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
                            self.worker_critic.model.set_weights(self.global_critic.model.get_weights())
                            # update global step
                            global_step += 1
                        # if episode is done
                        if done:
                            # update global episode count
                            global_episode_count += 1
                            # print episode rewards

                            print('Worker name: ', self.worker_name,
                                  ", Episode: ",  global_episode_count,
                                ', Step: ', step, ', Reward: ', episode_reward)
                            global_episode_reward.append(episode_reward)
                            # save episode reward at every 10th episodes
                            if global_episode_count % 10 == 0:
                                self.global_actor.save_weights("./save_weights/_actor.h5")
                                self.global_critic.save_weights("./save_weights/_critic.h5")


