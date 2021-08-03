#https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

class CartPoleSolver():
    def __init__(self):
        self.num_iterations = 2;0000

        self.inital_collect_steps = 100
        self.collect_steps_per_iteration = 1
        self.replay_buffer_max_length = 100000

        self.batch_size = 64
        self.learning_rate = 1e-3
        self.log_interval = 200

        self.num_eval_episodes = 10
        self.eval_interval = 1000
        
        self.fc_layer_params = (100, 50)

        self.create_env()
        self.create_network()
        self.create_agent()
        self.create_replay_buffer()

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=self.batch_size, 
            num_steps=2).prefetch(3)

        self.dataset
        self.iterator = iter(self.dataset)

        self.train_agent()
        self.plot()

    def create_env(self):
        env_name = 'CartPole-v0'
        self.env = suite_gym.load(env_name)
 
        train_py_env = suite_gym.load(env_name)
        eval_py_env = suite_gym.load(env_name)

        self.train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    def dense_layer(self, num_units):
        return tf.keras.layers.Dense(num_units, 
                                    activation = tf.keras.activations.relu,
                                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                                    scale=2.0, mode='fan_in', distribution='truncated_normal'))
   
    def create_network(self):
        self.action_tensor_spec = tensor_spec.from_spec(self.env.action_spec())
        self.num_actions = self.action_tensor_spec.maximum - self.action_tensor_spec.minimum + 1
        self.dense_layers = [self.dense_layer(num_units) for num_units in self.fc_layer_params]

        self.q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        self.q_net = sequential.Sequential(self.dense_layers + [self.q_values_layer])

    def create_agent(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=self.q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)
        
        self.agent.initialize()

    def create_replay_buffer(self):
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                                        data_spec=self.agent.collect_data_spec,
                                        batch_size=self.train_env.batch_size,
                                        max_length=self.replay_buffer_max_length)
    def collect_step(self, environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(self, env, policy, buffer, steps):
        for _ in range(steps):
            self.collect_step(env, policy, buffer)

    def train_agent(self):
        try:
            %%time
        except:
            pass
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self.agent.train = common.function(self.agent.train)

        # Reset the train step
        self.agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        self.avg_return = self.compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
        self.returns = [self.avg_return]

        for _ in range(self.num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            self.collect_data(self.train_env, self.agent.collect_policy, self.replay_buffer, self.collect_steps_per_iteration)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(self.iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg_return = self.compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                self.returns.append(avg_return)

    def compute_avg_return(self, environment, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
        
    def plot(self):
        iterations = range(0, self.num_iterations + 1, self.eval_interval)
        plt.plot(iterations, self.returns)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.ylim(top=250)
    '''
     def create_random_policy(self):
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy
        random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(),self.train_env.action_spec())
        example_environment = tf_py_environment.TFPyEnvironment(
        suite_gym.load('CartPole-v0'))
        time_step = example_environment.reset()
        random_policy.action(time_step)
    '''

if __name__=='__main__':
    cartpole = CartPoleSolver()