import gym
import dill
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from gym import wrappers
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error
from matplotlib import pyplot as plt
# re-organize immport

class AgentInfo:

    def __init__(self, 
                 dim_state,
                 n_actions,
                 q_function='FC', 
                 n_layer=3,
                 selection='egreedy',
                 epsilon=0.05,
                 epsilon_decay=1,
                 epsilon_min=0.01,
                 temp=1,
                 temp_decay=1,
                 alpha=0.0001,
                 gamma=0.99,
                 batch_size=64,
                 buffer_size=5000, 
                 experience_replay=False, 
                 target_network=False):
        """_summary_
            
        Args:
            dim_state (int): dimension of the state space
            n_actions (int): the number of total actions
            q_function (str, optional): eural network to be used (e.g. FC, CNN). 
                                        Defaults to 'none'.
            n_layer (int, optional): _description_. 
                                     Defaults to 3.
            selection (str, optional): _description_. 
                                       Defaults to 'egreedy'.
            epsilon (float, optional): _description_. 
                                       Defaults to 0.05.
            epsilon_decay (int, optional): _description_. 
                                           Defaults to 1.
            temp (int, optional): _description_. 
                                  Defaults to 1.
            temp_decay (int, optional): _description_. 
                                        Defaults to 1.
            alpha (float, optional): _description_. 
                                     Defaults to 0.001.
            gamma (float, optional): _description_. 
                                     Defaults to 0.99.
            batch_size (int, optional): size of the batch for experience replay. 
                                        Defaults to 32.
            buffer_size (int, optional): size of the memory buffer for experience replay. 
                                         Defaults to 2000.
            experience_replay (bool, optional): true to use experience replay. 
                                                Defaults to False.
            target_network (bool, optional): true to use target network. 
                                             Defaults to False.
        """
        self.dim_state = dim_state
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.selection = selection
        self.epsilon = epsilon # initial value
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # memory buffer for experience replay
        self.memory_buffer = []
        if experience_replay:
            self.buffer_size = buffer_size
            self.batch_size = batch_size
        else:
            self.buffer_size = 1
            self.batch_size = 1

        # construct q function (neural network)
        self.q_function = q_function
        self.n_layer = n_layer


class DQNAgent:

    def __init__(self,
                 agent_info=None,
                 build_model=False):
        """_summary_

        Args:
            agent_info (_type_, optional): _description_. Defaults to None.
            build_model (bool, optional): _description_. Defaults to False.
        """

        self.info = agent_info
        self.model = Sequential([])

        # need info to build
        # thus if we are loading agent
        # we don't build the model here
        if build_model:
            self.build_model()

        # TODO: grid search
        # TODO: target network

    def save(self, info_filename, model_path):
        info_file = open(info_filename, "wb")
        dill.dump(self.info, info_file)
        self.model.save(model_path)

    def load(self, info_filename, model_path):
        info_file = open(info_filename, "rb")
        self.info = dill.load(info_file)
        self.model = keras.models.load_model(model_path)

    def build_model(self):

        if self.info.q_function == 'FC':
            if (self.info.n_layer<2) or (type(self.info.n_layer) is not int):
                raise KeyError("n_layer must be integer and at least 2.")
            # the input layer
            self.model.add(
                Dense(16, input_dim=self.info.dim_state, activation='relu'))
            for i in range(self.info.n_layer-2):
                self.model.add(Dense(16, activation='relu'))
            # the output layer
            self.model.add(Dense(self.info.n_actions, activation='linear'))
        
        # TODO: other networks (e.g. CNN)

        # compilation
        self.model.compile(loss="mse",
                           optimizer=Adam(self.info.alpha))
        
    def select_action(self, current_state):
        """_summary_

        Args:
            current_state (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.info.selection == "egreedy":
            if np.random.uniform(0,1) < self.info.epsilon:
                action = np.random.choice(range(self.info.n_actions))
            else:
                q_values = self.model.predict(current_state)[0]
                action = np.argmax(q_values)
        
        if self.info.selection == 'boltzmann':
            q_values = self.model.predict(current_state)[0]
            probability = np.softmax(q_values, self.info.temp)
            cdf = np.cumsum(probability)
            action = np.where(np.random.uniform(0,1) <= cdf)[0][0]

        return action
      
    def update_exploration_probability(self):
        if self.info.selection == 'egreedy':
            if self.info.epsilon <= self.info.epsilon_min:
                pass
            else:
                self.info.epsilon = self.info.epsilon * self.info.epsilon_decay
        elif self.info.selection == 'boltzmann':
            self.info.temp = self.info.temp * self.info.temp_decay
    
    def store_episode(self,current_state, action, reward, next_state, done):
        """_summary_

        Args:
            current_state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            done (function): _description_
        """
        self.info.memory_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })

        # limit the memory buffer size
        # pop the oldest when exceed
        if len(self.info.memory_buffer) > self.info.buffer_size:
            self.info.memory_buffer.pop(0)
    
    def train(self):
        """_summary_
        """

        # TODO: change this
        # TODO: batch trainning

        _batch_size = min(len(self.info.memory_buffer), self.info.batch_size)
        batch_sample = np.random.choice(self.info.memory_buffer, _batch_size, replace=False)
        
        for experience in batch_sample:
            # We compute the Q-values of S_t
            q_current_state = self.model.predict(experience["current_state"])
            
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            
            if not experience["done"]:
                q_target = q_target + self.info.gamma*np.max(self.model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            
            # train the model
            self.model.fit(experience["current_state"], q_current_state, verbose=0)

        self.update_exploration_probability()

env = gym.make("CartPole-v1")
dim_state = env.observation_space.shape[0]
n_actions = env.action_space.n

n_episodes = 500
max_iteration_ep = 500

# no experience replay
agent_info = AgentInfo(dim_state, n_actions)

# with experience replay
agent_info = AgentInfo(dim_state, n_actions, epsilon=0.05, 
                       epsilon_decay=1.00, experience_replay=True)

agent = DQNAgent(agent_info, build_model=True)

total_reward = np.zeros(n_episodes)

for episode in range(n_episodes):
    current_state = env.reset()
    current_state = np.array([current_state])
    
    for step in range(max_iteration_ep):
        action = agent.select_action(current_state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.array([next_state])
        
        agent.store_episode(current_state, action, reward, next_state, done)
        total_reward[episode] += reward
        
        if done:
            break

        current_state = next_state

    agent.train()

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(len(total_reward)), total_reward)
ax.set_xlabel("episode number")
ax.set_ylabel("reward")
plt.savefig("curve.pdf")

agent.save(info_filename="info.pkl", model_path="model")