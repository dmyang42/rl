#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

from black import re
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
from tqdm import tqdm

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        # means Q(s,a) to 0
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        a = np.argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        self.Q_sa[s, a] = 0
        for s_next in range(self.n_states):
            self.Q_sa[s, a] += p_sas[s, a, s_next] * \
                (r_sas[s, a, s_next] + self.gamma * np.max(self.Q_sa[s_next])) 
        return True    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
        
    max_error = threshold
    i = 0
    while(max_error >= threshold):
        max_error = 0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                x = QIagent.Q_sa[s, a]
                QIagent.update(s, a, env.p_sas, env.r_sas)
                max_error = max(max_error, np.abs(x - QIagent.Q_sa[s, a]))   
    
        # Plot current Q-value estimates & print max error
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True,
                   step_pause=0.2)
        title = "Dynamic Programming - Iteration: {}".format(i)
        # env.snapshot(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,
        #              setup_name="dynamic_programming/dynamic_programming",iteration=i,title=title)
        print("Q-value iteration, iteration {}, max error {}".format(i, max_error))
        i = i + 1
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    # env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)

    # View optimal policy

    done = False
    s = env.reset()
    reward = 0
    step_count = 0
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

        reward = reward + r
        step_count = step_count + 1

    mean_reward_per_timestep = reward / step_count
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

if __name__ == '__main__':
    experiment()
