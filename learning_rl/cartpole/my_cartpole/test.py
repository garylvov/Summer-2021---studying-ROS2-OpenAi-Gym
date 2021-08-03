#https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
import gym
import numpy as np
import tensorflow as tf
from tf_agents import agents
from tf_agents.specs import tensor_spec
from tf_agents.environments import tf_py_environment

class CartPoleSolver():
    def __init__(self):
        self.create_env()

    def create_env(self):
        self.env = gym.make('CartPole-v0')
        print(self.env.action_space)
        print(self.env.action_space.n)
        print(self.env.observation_space)
        print(self.env.observation_space.shape[0])
        print(self.env.reward_range)
if __name__=='__main__':
    my_cartpole = CartPoleSolver()
