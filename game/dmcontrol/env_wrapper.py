import numpy as np

class DMCWrapper:
    def __init__(self, env, reward_func=None):
        """

        :param env: instance of gym environment
        :param k: no. of observations to stack
        """
        self.env = env
        # print(self.env.env)
        self.action_space = env.action_space
        self.reward_func = reward_func

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.reward_func is not None:
            reward = self.reward_func.reward(self.env.env)
        observation = np.asarray(observation, dtype="float32") / 255.0
        observation = np.moveaxis(observation, -1, 0)
        return observation, reward, done, info

    def reset(self, **kwargs):
        # print("MAIN RESET")
        observation = self.env.reset(**kwargs)
        if self.reward_func is not None:
            self.reward_func.reset(0)

        observation = np.asarray(observation, dtype="float32") / 255.0
        observation = np.moveaxis(observation, -1, 0)
        # print("MAIN RESET RETURN")
        return observation

    def close(self):
        self.env.close()


class DMCLowDimWrapper:
    def __init__(self, env, reward_func=None):
        """

        :param env: instance of gym environment
        :param k: no. of observations to stack
        """
        self.env = env
        # print(self.env.env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_func = reward_func

    def step(self, action):
        # print('Executing Action', action)
        observation, reward, done, info = self.env.step(action)
        if self.reward_func is not None:
            reward = self.reward_func.reward(self.env.env)
        observation = np.asarray(observation, dtype="float32")
        return observation, reward, done, info

    def reset(self, **kwargs):
        # print("MAIN RESET")
        observation = self.env.reset(**kwargs)
        if self.reward_func is not None:
            self.reward_func.reset(0)

        observation = np.asarray(observation, dtype="float32")
        return observation

    def close(self):
        self.env.close()
