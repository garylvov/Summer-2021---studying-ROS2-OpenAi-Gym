# A more object oriented approach of /Nicholas-Renotte-Tutorial/first_cartpole.py
import gym
import random
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

class CartPoleSolver():
    def __init__(self, policy=BoltzmannQPolicy()):
        #enviroment
        self.env = gym.make('CartPole-v0')
        self.states = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n
        
        #model
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,self.states)))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(48, activation='relu'))
        self.model.add(Dense(self.actions, activation='linear'))
        self.model.summary()

        #memory
        self.policy = policy
        self.memory = SequentialMemory(limit=5000, window_length=1)
        self.dqn = DQNAgent(model=self.model, memory=self.memory, policy = self.policy,
                            nb_actions = self.actions, nb_steps_warmup=10, target_model_update=1e-2)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def run(self):
        self.dqn.fit(self.env, nb_steps=50000, visualize=False, verbose=1)
        scores = self.dqn.test(self.env, nb_episodes = 100, visualize=False)

        scores = self.dqn.test(self.env, nb_episodes = 100, visualize=False)
        print(np.mean(scores.history['episode_reward']))

        _ = self.dqn.test(self.env, nb_episodes=15, visualize=True)
        self.dqn.save_weights('dqn_weights.h5f', overwrite=True)
        self.env.close() 

if __name__=='__main__':
    my_cartpole = CartPoleSolver()
    my_cartpole.run()