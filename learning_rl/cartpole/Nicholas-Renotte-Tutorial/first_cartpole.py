#https://www.youtube.com/watch?v=cO5g5qLrLSo&t=850s
import gym
import random
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    states = env.observation_space.shape[0]
    actions = env.action_space.n

    model = build_model(states, actions)
    model.summary()

    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    scores = dqn.test(env, nb_episodes = 100, visualize=False)
    print(np.mean(scores.history['episode_reward']))

    _ = dqn.test(env, nb_episodes=15, visualize=True)
    dqn.save_weights('dqn_weights.h5f', overwrite=True)

    env.close()