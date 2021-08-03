#Decreased amount of neurons in model, & episodes in visualized demos
#Switched Graphs to be scores vs episodes instead of scores vs steps
#Started to play around with graphing more metrics, constrained by metrics avialable in keras rl lib
#   Can't use tensorboard when mixing keras rl with tensorflow keras
#       Going to switch to using tensorflow DQN_agent instead of rl.agents DQNAgent 
import gym
import random
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

class CartPoleSolver():
    def __init__(self, policy=BoltzmannQPolicy()):
        self.create_env()
        self.create_model()
        self.create_memory(policy)

    def create_env(self):
        self.env = gym.make('CartPole-v0')
        self.states = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n

    def create_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,self.states)))
        self.model.add(Dense(12, activation='relu'))
        self.model.add(Dense(28, activation='relu'))
        self.model.add(Dense(self.actions, activation='linear'))
        self.model.summary()

    def create_memory(self, policy):
        self.policy = policy
        self.memory = SequentialMemory(limit=5000, window_length=1)
        self.dqn = DQNAgent(model=self.model, memory=self.memory, policy = self.policy,
                            nb_actions = self.actions, nb_steps_warmup=10, target_model_update=1e-2)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def train_model(self):
        print("\n\n\n\nTRAINING MODEL\n\n\n\n")

        history = self.dqn.fit(self.env, nb_steps=50000, visualize=True, verbose=1)
        print(history.history.keys())
        
        self.scores = (history.history['episode_reward'])
        self.episodes = [None] * len(self.scores)
        '''
        self.loss = (history.history['loss'])
        self.mae = (history.history['mae'])
        self.mean_q = (history.history['mean_q'])
        '''
        scores = self.dqn.test(self.env, nb_episodes = 100, visualize=False)
        print(np.mean(scores.history['episode_reward']))

        _ = self.dqn.test(self.env, nb_episodes=5, visualize=True)
        self.dqn.save_weights('dqn_weights.h5f', overwrite=True)
        self.env.close()

    def plot(self):
        print("\n\n\n\nPLOT\n\n\n\n\n\n")
        #plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Score')

        for x in range (len(self.scores)):
            self.episodes[x] = x
        scores_line = plt.plot(self.episodes, self.scores)
        plt.setp(scores_line,'color', 'r', 'linewidth', 2.0)
        '''
        plt.subplot()
        plt.xlabel('Episode')
        loss_line = plt.plot(self.loss, self.scores)
        mae_line = plt.plot(self.mae, self.scores)
        mean_q_line = plt.plot(self.mean_q, self.scores)

        plt.setp(loss_line,'color', 'g', 'linewidth', 2.0)
        plt.setp(mae_line,'
        color', 'b', 'linewidth', 2.0)
        plt.setp(mean_q_line,'color', 'o', 'linewidth', 2.0)
        plt.legend()
        '''
        plt.show()

    def load_weights(self):
        self.dqn.load_weights('dqn_weights.h5f')
        _ = self.dqn.test(self.env, nb_episodes=10, visualize=True)
        self.env.close() 

    def large_test(self):
        test = self.dqn.test(self.env, nb_episodes=100, visualize=False)
        print(test.history.keys())
        self.scores = (test.history['episode_reward'])
        self.episodes = (test.history['nb_steps'])
        print(len(self.scores))
        print(len(self.episodes))
        
    def reset_model(self):
        del self.model
        del self.dqn
        del self.env
        self.create_env()
        self.create_model()
        self.create_memory()

if __name__=='__main__':
    my_cartpole = CartPoleSolver()
    my_cartpole.train_model()
    my_cartpole.plot()

    loaded_weights_cartpole = CartPoleSolver()
    loaded_weights_cartpole.load_weights()