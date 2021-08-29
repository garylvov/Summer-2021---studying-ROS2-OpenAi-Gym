import array_escape
import gym
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

class DQN():
    def __init__(self):
        self.create_env()
        self.create_model()
        self.create_memory()

    def create_env(self):
        self.env = gym.make('array_escape-v2')
        self.states = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n
        
    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, 1, strides=8, input_shape=(1, self.states, self.states), activation="relu"))
        self.model.add(Conv2D(64, 1, strides=4, activation="relu"))
        self.model.add(Conv2D(64, 1, strides=3, activation="relu"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(self.actions, activation='linear'))
        self.model.summary()

    def create_memory(self):
        self.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=.1, value_test=.2, nb_steps=1000000)
        self.memory = SequentialMemory(limit=10000000, window_length=1)
        self.dqn = DQNAgent(model=self.model, memory=self.memory, policy = self.policy,
                            nb_actions = self.actions, nb_steps_warmup=100, target_model_update=1e-2)
        self.dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    def train_model(self):
        print("\n\n\n\n\n\n\n\n")
        history = self.dqn.fit(self.env, nb_steps=1000000, visualize=False, verbose=1)
        
        self.scores = (history.history['episode_reward'])
        self.episodes = [None] * len(self.scores)

        scores = self.dqn.test(self.env, nb_episodes = 100, visualize=False)
        print(np.mean(scores.history['episode_reward']))

        _ = self.dqn.test(self.env, nb_episodes=5, visualize=True)
        self.dqn.save_weights('dqn_weights.h5f', overwrite=True)
        self.env.close()

    def plot(self):
        print("\n\n\n\n\n\n\n\n")
        plt.xlabel('Episode')
        plt.ylabel('Score')

        for x in range (len(self.scores)):
            self.episodes[x] = x
        scores_line = plt.plot(self.episodes, self.scores)

        plt.setp(scores_line,'color', 'r', 'linewidth', 2.0)
        plt.show()

    def load_weights(self):
        self.dqn.load_weights('dqn_weights.h5f')
        _ = self.dqn.test(self.env, nb_episodes=10, visualize=True)
        self.env.close() 

    def large_test(self):
        test = self.dqn.test(self.env, nb_episodes=250, visualize=True)
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
    model = DQN()
    model.train_model()
    model.plot()
    
    loaded_weights_model = DQN()
    loaded_weights_model.load_weights()
    loaded_weights_model.large_test()