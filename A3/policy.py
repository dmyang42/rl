'''
 # @ Author: Yi Kang, Simin Tong, Daming Yang @ Leiden
 # @ Create Time: 2222-22-22 00:00:00
 # @ Modified time: ????-??-?? ??:??:??
 # @ Description: REINFOCE / Actor-Critic (with / without baseline) on CartPole-v1
'''

'''
    >>> python policy.py {ALGORITHM} --file-name {OUTPUT} --episode {NUMBER OF EPISODE}
'''
import shutup
shutup.please()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from IPython import embed
from tensorflow import keras
from torch import constant_pad_nd
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import utils as np_utils
from tensorflow.keras import optimizers

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Agent(object):

    def __init__(self, 
                 state_dim, 
                 n_actions, 
                 hidden_neurons=[32, 32],
                 discount_rate=0.99,
                 algorithm="REINFORCE",
                 n_step=10,
                 episodic=False):

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.discount_rate = discount_rate
        self.algorithm = algorithm
        self.n_step = n_step

        self.PiNet = self.__build_network(state_dim, n_actions, hidden_neurons, output_activation="softmax")
        self.__compile_PiNet() # manual compile PiNet
        self.VlNet = self.__build_network(state_dim, output_dim=1, hidden_dims=[32,32], output_activation="relu")
        self.VlNet.compile(loss="mse", optimizer=optimizers.Adam(learning_rate=1e-2))

        self.episodic = episodic

    def __build_network(self, input_dim, output_dim, hidden_dims=[32,32], output_activation="relu"):
        """
            Create a base network
        """
        self.X = layers.Input(shape=(input_dim,))
        net = self.X

        for h_dim in hidden_dims:
            net = layers.Dense(h_dim,
                               kernel_initializer='random_normal')(net)
            net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim,
                           kernel_initializer='random_normal')(net)
        net = layers.Activation(output_activation)(net)

        return Model(inputs=self.X, outputs=net)

    def __compile_PiNet(self):
        """
            Create a fit function for PiNet and compile
        """
        action_prob_placeholder = self.PiNet.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.n_actions),
                                                  name="action_onehot")
        delta_placeholder = K.placeholder(shape=(None,),
                                          name="discount_reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss_PiNet = - log_action_prob * delta_placeholder
        loss_PiNet = K.mean(loss_PiNet)

        adam = optimizers.Adam(learning_rate=2e-3)

        updates_PiNet = adam.get_updates(params=self.PiNet.trainable_weights,
                                         loss=loss_PiNet)

        self.train_PiNet = K.function(inputs=[self.PiNet.input,
                                              action_onehot_placeholder,
                                              delta_placeholder],
                                      outputs=[self.PiNet.output],
                                      updates=updates_PiNet)

    def select_action(self, state):
        """Returns an action at given `state`
        Args:
            state (1-D or 2-D Array): It can be either 1-D array of shape (state_dimension, )
                or 2-D array shape of (n_samples, state_dimension)
        Returns:
            action: an integer action value ranging from 0 to (n_actions - 1)
        """
        shape = state.shape

        if len(shape) == 1:
            assert shape == (self.state_dim,), "{} != {}".format(shape, self.state_dim)
            state = np.expand_dims(state, axis=0)

        elif len(shape) == 2:
            assert shape[1] == (self.state_dim), "{} != {}".format(shape, self.state_dim)

        else:
            raise TypeError("Wrong state shape is given: {}".format(state.shape))

        action_prob = np.squeeze(self.PiNet.predict(state))
        assert len(action_prob) == self.n_actions, "{} != {}".format(len(action_prob), self.n_actions)
        
        return np.random.choice(np.arange(self.n_actions), p=action_prob)

    def fit(self, S, A, R):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """
        action_onehot = np_utils.to_categorical(A, num_classes=self.n_actions)
        discount_reward = compute_discounted_R(R, episodic=self.episodic)

        if self.algorithm == "REINFORCE":
            delta = discount_reward[:-1]

        elif self.algorithm == "baseline":
            delta = (discount_reward - self.VlNet.predict(S).flatten())[:-1]
            target_Vl = discount_reward[:-1]

        elif self.algorithm == "bootstrapping":
            q_t = R[:-1]
            n = self.n_step
            for i in range(1, min(n,len(q_t))):
                q_t = q_t + (self.discount_rate**i) * np.pad(R[i:-1], (0,i))
            if len(q_t)>n:
                bootstrap_value = np.pad(self.VlNet.predict(S[n:]).flatten(), (0,n-1))
            else:
                bootstrap_value = 0
            q_t = q_t + self.discount_rate**n * bootstrap_value

            delta = q_t
            target_Vl = q_t

        elif self.algorithm == "baseline+bootstrapping":
            q_t = R[:-1]
            n = self.n_step
            for i in range(1, min(n,len(q_t))):
                q_t = q_t + (self.discount_rate**i) * np.pad(R[i:-1], (0,i))
            if len(q_t)>n:
                bootstrap_value = np.pad(self.VlNet.predict(S[n:]).flatten(), (0,n-1))
            else:
                bootstrap_value = 0
            q_t = q_t + self.discount_rate**n * bootstrap_value
            
            delta = q_t - self.VlNet.predict(S[:-1]).flatten()
            target_Vl = q_t

        if delta.std() == 0:
            print(f"{bcolors.FAIL}{bcolors.BOLD}>>> {bcolors.ENDC}Value network pooping shit!{bcolors.ENDC}")
            pass
        else:
            delta = (delta - delta.mean()) / delta.std()

        if self.algorithm != "REINFORCE":
            self.VlNet.fit(S[:-1], target_Vl, verbose=False)
        self.train_PiNet([S[:-1], action_onehot[:-1], delta])

def compute_discounted_R(R, discount_rate=.99, episodic=False):
    """Returns discounted rewards
    Args:
        R (1-D array): a list of `reward` at each time step
        discount_rate (float): Will discount the future value by this rate
    Returns:
        discounted_r (1-D array): same shape as input `R`
            but the values are discounted
    Examples:
        >>> R = [1, 1, 1]
        >>> compute_discounted_R(R, .99) # before normalization
        [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
    """
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):

        running_add = running_add * discount_rate + R[t]

        if episodic:
            discounted_r[t] = discount_rate**t * running_add
        else:
            discounted_r[t] = running_add

    return discounted_r

def run_episode(env, agent):
    done = False
    S, A, R = [], [], []

    s = env.reset()
    total_reward = 0

    while not done:
        a = agent.select_action(s)
        s2, r, done, _ = env.step(a)
        total_reward = total_reward + r

        S.append(s)
        A.append(a)
        R.append(r)

        s = s2

        if done:
            S, A, R = np.array(S), np.array(A), np.array(R)
            agent.fit(S, A, R)

    return total_reward


def run(n_episode, algorithm, n_step, episodic, path, file_name):
    try: # for ctrl+C
        print(f"{bcolors.OKGREEN}{bcolors.BOLD}>>> {bcolors.ENDC}Set up environment {bcolors.BOLD}{bcolors.OKCYAN}CartPole-v1{bcolors.ENDC}")
        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # hard code
        hidden_neurons = [32, 32]
        discount_rate = 0.99

        print(f"{bcolors.OKGREEN}{bcolors.BOLD}>>> {bcolors.ENDC}Initialize agent with algorithm {bcolors.BOLD}{bcolors.OKCYAN}{algorithm}{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}{bcolors.BOLD}>>> {bcolors.ENDC}Policy network hidden_neurons={hidden_neurons}, discount rate={discount_rate}, n step={n_step}, episodic={episodic}{bcolors.ENDC}")
        agent = Agent(state_dim, n_actions, hidden_neurons, discount_rate=discount_rate, algorithm=algorithm, n_step=n_step, episodic=episodic)

        print(f"{bcolors.OKGREEN}{bcolors.BOLD}>>> {bcolors.OKCYAN}Policy network:{bcolors.ENDC}")
        print(agent.PiNet.summary())

        print(f"{bcolors.OKGREEN}{bcolors.BOLD}>>> {bcolors.OKCYAN}Value network:{bcolors.ENDC}")
        print(agent.VlNet.summary())

        rewards = np.zeros(n_episode)

        print(f"{bcolors.OKGREEN}{bcolors.BOLD}>>> {bcolors.ENDC}Start trainning{bcolors.ENDC}")
        for episode in tqdm(range(n_episode), desc='Episode'):
            reward = run_episode(env, agent)
            #print(episode, reward)
            rewards[episode] = reward

        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f"{path}/{file_name}.npy", rewards)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(np.arange(n_episode), rewards)
        ax.set_xlabel('Episode Number')
        ax.set_ylabel('Reward')
        print(f"{bcolors.OKGREEN}{bcolors.BOLD}>>> {bcolors.ENDC}Save learning curve to {bcolors.BOLD}{bcolors.OKCYAN}{path}/{file_name}.pdf{bcolors.ENDC}")
        plt.savefig(f'{path}/{file_name}.pdf')
        plt.close()

    except KeyboardInterrupt:
        print(f"{bcolors.FAIL}{bcolors.BOLD}>>> {bcolors.ENDC}Aboard trainning...{bcolors.ENDC}")
        env.close()

    # NOTE You can use embed and the following code to see the trained CartPole!
    # embed()

"""
env = gym.make("CartPole-v1")
for i_episode in range(10):
    state = env.reset()
    for t in range(500):
        state = np.array([state])
        env.render()
        action = agent.select_action(state)                        
        state, reward, done, info = env.step(action)
        if done:
            break
    print("Episode finished after {} timesteps".format(t+1))
env.close()
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Perform REINFOCE / Actor-Critic (with / without baseline) on Cartpole-v1")
    parser.add_argument("algorithm", choices=["REINFORCE", "baseline", "bootstrapping", "baseline+bootstrapping"], 
                        help="available: REINFORCE / baseline / bootstrapping / baseline+bootstrapping")
    parser.add_argument("--episode", type=int, default=2000, help="number of episodes for trainning")
    parser.add_argument("--TD-step", type=int, default=10, help="Temporal different n-step for bootstrapping")
    parser.add_argument("--path", default=".", help="path name")
    parser.add_argument("--file-name", default="learning_curve", help="file name of learning curve plot")
    parser.add_argument("--episodic", action="store_true", help="episodic or not; for REINFORCE")
    args = parser.parse_args()
    run(args.episode, args.algorithm, args.TD_step, args.episodic, args.path, args.file_name)
