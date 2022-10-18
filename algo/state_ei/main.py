import copy
import math
import os

# import nevergrad
import numpy
import ray
import torch

# import diagnose_model
from algo.state_ei.model import MLPModel
from algo.state_ei.trainer import Trainer
from algo.state_ei.replay_buffer import ReplayBuffer
from algo.state_ei.player import PlayerWorker, Player
from algo.state_ei.batch_worker import BatchWorker, BatchBufferFast
from algo.state_ei.parallel_model import RZeroMLPWorker

import shared_storage

from pickle_utils import *


class RZero:

    def __init__(self, config=None, test_throughput=False):
        # Load the game and the config from the module with the game name
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        print('Start to init ray.')
        ray.init(num_gpus=config.num_gpus,
                 num_cpus=config.total_cpus,
                 object_store_memory=150 * 1024 * 1024 * 1024,
                 ignore_reinit_error=True)
        print('Finished Ray Init')
        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "target_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
            "force_selfplay_halt": False
        }

        self.replay_buffer = {}# load_data('replay.pkl')
        self.test_throughput = test_throughput

        if test_throughput:
            self.replay_buffer = load_data('replay.pkl')
            self.checkpoint["num_played_games"] = 320
            self.checkpoint["num_played_steps"] = int(320 * 250)
            self.checkpoint["force_selfplay_halt"] = True

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))
        self.checkpoint["target_weights"] = copy.deepcopy(self.checkpoint["weights"])
        self.checkpoint["selfplay_weights"] = copy.deepcopy(self.checkpoint["weights"])

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage = None

    def train_new(self, log_in_tensorboard):
        print("Start training!")
        if log_in_tensorboard or self.config.save_model:
            os.makedirs(self.config.results_path, exist_ok=True)

        # Initialize workers
        # Workers: Training worker
        self.training_worker = Trainer(self.checkpoint, self.config)

        # Parameter server.
        self.shared_storage = shared_storage.SharedStorage.remote(
            self.checkpoint, self.config,
        )
        self.shared_storage.set_info.remote("terminate", False)
        self.shared_storage.set_info.remote("running_games", False)

        # Workers: Replay buffer workers.
        self.replay_buffer_worker = ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config, self.test_throughput
        )

        # Worker: Batch Buffer
        self.batch_buffer_worker = BatchBufferFast()

        self.batch_workers = [
            BatchWorker.remote(
                idx, self.checkpoint, self.batch_buffer_worker, self.replay_buffer_worker,
                self.shared_storage, self.config
            )
            for idx in range(self.config.batch_worker_num)
        ]

        [
            batch_worker.spin.remote()
            for batch_worker in self.batch_workers
        ]

        # if log_in_tensorboard:
        self.training_worker.continuous_update_weights(
            self.batch_buffer_worker, self.replay_buffer_worker, self.batch_workers, self.shared_storage
        )

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage:
            self.shared_storage.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage = None

    def test_new(self, model_path, num_tests=1):
        model_state_dict = torch.load(model_path, map_location="cpu")
        player = Player(self.config, 42, n_parallel=num_tests)
        player.model.load_state_dict(model_state_dict)
        all_frames = player.run_render()
        return all_frames

    def test(
            self, model_path, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """

        self.shared_storage = shared_storage.SharedStorage.remote(
            self.checkpoint, self.config,
        )

        model_state_dict = torch.load(model_path, map_location="cpu")

        self.shared_storage.set_info.remote({"weights": copy.deepcopy(model_state_dict)})
        self.model_workers = [
            RZeroMLPWorker.remote(self.config)
            for _ in range(2)
        ]

        test_results = []
        for i in range(num_tests):
            play_worker = PlayerWorker.remote(self.config, self.model_workers[i // 4], 1000 + i)

            test_result = play_worker.test.remote(self.shared_storage)
            test_results.append(test_result)

        all_results = ray.get(test_results)

        results = []
        for res in all_results:
            print("Reward", sum(res.reward_true_history))
            results.append(sum(res.reward_true_history))

        mean_return = sum(results) / len(results)
        print("MEAN RETURN = ", mean_return)
        return

    def test_full(
            self, model_path, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """

        self.shared_storage = shared_storage.SharedStorage.remote(
            self.checkpoint, self.config,
        )

        model_state_dict = torch.load(model_path, map_location="cpu")

        self.shared_storage.set_info.remote({"weights": copy.deepcopy(model_state_dict)})
        self.model_workers = [
            RZeroMLPWorker.remote(self.config)
            for _ in range(2)
        ]

        test_results = []
        for i in range(num_tests):
            self_play_worker = PlayerWorker.remote(self.config, self.model_workers[i // 4], 1000 + i)

            test_result = self_play_worker.test.remote(self.shared_storage)
            test_results.append(test_result)

        all_results = ray.get(test_results)

        rewards = []

        all_rollouts = []
        for res in all_results:
            child_qinits = res.child_qinits
            child_values = res.child_values
            child_visits = res.child_visits
            actions = res.root_actions[1:]
            frames = res.frames

            clip_length = min([len(child_qinits), len(child_values),
                               len(child_visits), len(actions), len(res.observation_history)])

            rollout = {'tree': res.trees[:clip_length],
                       'phys': res.phys_states[:clip_length],
                       'frames': frames[:clip_length],
                       'child_q': child_qinits[:clip_length],
                       'child_val': child_values[:clip_length],
                       'child_vis': child_visits[:clip_length],
                       'actions': actions[:clip_length],
                       'obs': res.observation_history[:clip_length]}

            all_rollouts.append(rollout)
            print("Reward", sum(res.reward_true_history))
            rewards.append(sum(res.reward_true_history))

        mean_return = sum(rewards) / len(rewards)
        print("MEAN RETURN = ", mean_return)
        return all_rollouts

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self.checkpoint = torch.load(checkpoint_path)
                print(f"\nUsing checkpoint from {checkpoint_path}")
            else:
                print(f"\nThere is no model saved in {checkpoint_path}.")

        # Load replay buffer
        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                with open(replay_buffer_path, "rb") as f:
                    replay_buffer_infos = pickle.load(f)
                self.replay_buffer = replay_buffer_infos["buffer"]
                self.checkpoint["num_played_steps"] = replay_buffer_infos[
                    "num_played_steps"
                ]
                self.checkpoint["num_played_games"] = replay_buffer_infos[
                    "num_played_games"
                ]
                self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                    "num_reanalysed_games"
                ]

                print(f"\nInitializing replay buffer with {replay_buffer_path}")
            else:
                print(
                    f"Warning: Replay buffer path '{replay_buffer_path}' doesn't exist.  Using empty buffer."
                )
                self.checkpoint["training_step"] = 0
                self.checkpoint["num_played_steps"] = 0
                self.checkpoint["num_played_games"] = 0
                self.checkpoint["num_reanalysed_games"] = 0


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = MLPModel(config)
        weights = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weights, summary

