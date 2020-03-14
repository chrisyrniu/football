import gym
import gfootball.env as grf_env
import numpy as np

class MultiAgentEnv(gym.Env):

	def __init__(self, num_agents):
		self.env = grf_env.create_environment(
			env_name='test_example_multiagent',
			stacked=False,
			representation='multiagent',
			rewards='scoring',
			write_goal_dumps=False,
			write_full_episode_dumps=False,
			render=False,
			dump_frequency=0,
			logdir='/tmp/rllib_test',
			extra_players=None,
			number_of_left_players_agent_controls=num_agents,
			number_of_right_players_agent_controls=0,
			channel_dimensions=(3, 3)
			)
		self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
		self.observation_space = gym.spaces.Box(
	        low=self.env.observation_space.low[0],
	        high=self.env.observation_space.high[0],
			dtype=self.env.observation_space.dtype)
		self.num_agents = num_agents

	def reset(self):
		original_obs = self.env.reset()
		# temp = np.array(original_obs)
		# print(temp)
		# for i in range(11):
		# 	print(temp[i,:,:,0])
		# 	print(temp[i,:,:,1])
		# 	print(temp[i,:,:,2])
		# 	print(temp[i,:,:,3])
		# 	print('\n')
		obs = {}
		for x in range(self.num_agents):
			if self.num_agents > 1:
				obs['agent_%d' % x] = original_obs[x]
			else:
				obs['agent_%d' % x] = original_obs
		return obs

	def step(self, action_dict):
		actions = []
		for key, value in sorted(action_dict.items()):
			actions.append(value)
		o, r, d, i = self.env.step(actions)
		rewards = {}
		obs = {}
		infos = {}
		for pos, key in enumerate(sorted(action_dict.keys())):
			infos[key] = i
			if self.num_agents > 1:
				rewards[key] = r[pos]
				obs[key] = o[pos]
			else:
				rewards[key] = r
				obs[key] = o
		dones = {'__all__': d}
		return obs, rewards, dones, infos

if __name__ == '__main__':
	test_env = MultiAgentEnv(3)
	test_env.reset()

