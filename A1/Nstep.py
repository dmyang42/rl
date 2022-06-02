#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

from math import gamma
from typing import Type
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax
from tqdm import tqdm

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            
            r = np.random.rand()
            if r < epsilon:
                # exlore
                a = np.random.choice(range(self.n_actions))
            else:
                # exploit
                a = np.argmax(self.Q_sa[s])            
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            
            # change to np.float128 for large exponential
            exponent = np.float128(self.Q_sa[s] / temp)
            denominator = np.sum(np.exp(exponent))
            prob_of_each_state = np.exp(exponent)/denominator
            prob_of_each_state = np.cumsum(prob_of_each_state)
            r = np.random.rand()
            
            # find the first (r - a) < 0 index
            a = np.where((r - prob_of_each_state) < 0)[0][0]

        return a

        
    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T_ep = len(states) - 1

        for t in range(T_ep):
            Gt = 0
            m = min(self.n, T_ep-t)
            for i in range(m):
                Gt = Gt + self.gamma**i * rewards[t+i]
            if done and (t+m == T_ep): # is terminal
                pass
            else:
                Gt = Gt + self.gamma**m * np.max(self.Q_sa[states[t+m]])

            self.Q_sa[states[t], actions[t]] = self.Q_sa[states[t], actions[t]] + self.learning_rate * (Gt - self.Q_sa[states[t], actions[t]])

        return True


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    rewards = []
    done = False

    for itr in range(n_timesteps):    
        s = env.reset()
        # collect episode
        states_ep, actions_ep, rewards_ep = [], [], []
        states_ep.append(s)
        for t in range(max_episode_length):

            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)
            states_ep.append(s_next)
            actions_ep.append(a)
            rewards_ep.append(r)
            if done:
                break
            else:
                s = s_next
        rewards.append(np.mean(rewards_ep))

        # compute n-step targets and update
        pi.update(states_ep, actions_ep, rewards_ep, done)

    if plot:
        env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=5) # Plot the Q-value estimates during n-step Q-learning execution


    return rewards 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(rewards))    
    
if __name__ == '__main__':
    test()
