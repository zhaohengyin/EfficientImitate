import copy
import os

import ray
import torch


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)
        self.timestamp_checkpoint = {}

    def add_timestamp_info(self, key, val, t):
        if key not in self.timestamp_checkpoint:
            self.timestamp_checkpoint[key] = []

        self.timestamp_checkpoint[key].append((int(t), val))

    def get_timestamp_info(self, keys):
        if isinstance(keys, str):
            if keys not in self.timestamp_checkpoint:
                return None
            return self.timestamp_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.timestamp_checkpoint[key] for key in keys if key in self.current_checkpoint}
        else:
            raise TypeError

    def save_checkpoint(self, path=None):
        if not path:
            path = os.path.join(self.config.results_path, "model.checkpoint")

        torch.save(self.current_checkpoint, path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            if keys not in self.current_checkpoint:
                return None
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys if key in self.current_checkpoint}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

    def inc_counter(self, key, delta=1):
        if isinstance(key, str):
            if key not in self.current_checkpoint:
                self.current_checkpoint[key] = 0
            # print("Inc")
            self.current_checkpoint[key] += delta

