import array_escape
import gym
import time

def random_agent(episodes=100000):
	env = gym.make("array_escape-v2")
	env.reset()
	env.render()
	for e in range(episodes):
		action = env.action_space.sample()
		state, reward, done, _ = env.step(action)
		
		env.render()
		print(reward)
		if done:
			env.reset()


if __name__ == "__main__":
    random_agent()