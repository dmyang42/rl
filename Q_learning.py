#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class QLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
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
        
    def update(self,s,a,r,s_next,done):
        ''' Function updates Q(s,a) '''
        Gt = r + self.gamma * np.max(self.Q_sa[s_next])
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (Gt - self.Q_sa[s, a])

        # maybe remove this?
        if done: # if s_next is terminal       
           self.Q_sa[s_next] = np.zeros(self.n_actions)

        return True

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    s = env.reset()
    done = False
    
    for t in range(n_timesteps):
        
        a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
        s_next, r, done = env.step(a)
        pi.update(s, a, r, s_next, done)
        
        if done:
            s = env.reset()
        else:
            s = s_next
        rewards.append(r)
        if plot:
            # Plot the Q-value estimates during Q-learning execution
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) 
    return rewards 

def test():
    
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 1

    # Exploration
    policy = 'softmax' # 'egreedy' or 'softmax' 
    epsilon = 0.2
    temp = 1.0
    
    # Plotting parameters
    plot = False

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))

if __name__ == '__main__':
    test()
