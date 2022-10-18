import numpy as np
import ray
from algo.obs_ei.model import MuZeroResidualNetwork
import torch_utils


@ray.remote(num_gpus=0.1)
class RZeroMLPWorker:
    def __init__(self, config):
        super().__init__()
        self.network = MuZeroResidualNetwork(config)
        self.network = self.network.cuda()
        self.network.eval()
        self.locked = 0

    def lock(self):
        self.locked += 1

    def unlock(self):
        self.locked -= 1

    def get_weights(self):
        return self.network.get_weights()

    def set_weights(self, weights):
        assert self.locked >= 0
        if self.locked <= 0:
            self.network.set_weights(weights)

    def eval_q(self, obs, actions):
        return self.network.eval_q(obs, actions)

    def sample_actions(self, policy, n):
        return torch_utils.tensor_to_numpy(self.network.sample_actions(policy, n))

    def sample_mixed_actions(self, policy, config, is_root):
        return self.network.sample_mixed_actions(policy, config, is_root)

    def initial_inference(self, observation):
        return self.network.initial_inference(observation)

    def recurrent_inference(self, encoded_state, action):
        return self.network.recurrent_inference(encoded_state, action)
