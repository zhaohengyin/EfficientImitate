import copy
import numpy
import numpy as np
import ray
import torch
import torch_utils
import time
from ray.util.queue import Queue
from algo.state_ei.trainer import MLPModel
from mcts_tree_sample.mcts import run_multi_gail_support


def calculate_mean_std(val, visit):
    '''

    :param val: [B, N, VAL]
    :param visit: [B, N]. The last dim is the visit count of each value.
    :return:
    '''
    # print(val.shape, visit.shape)
    visit = visit / np.sum(visit, axis=-1, keepdims=True)
    visit = visit.reshape(*visit.shape, -1)

    val_mean = np.sum(visit * val, axis=-2)
    val_mean_result = np.array(val_mean)

    val_mean = val_mean.reshape((val_mean.shape[0], 1, val_mean.shape[-1]))
    val_mean = val_mean.repeat(val.shape[1], axis=1)

    val_dev = (val_mean - val) * (val_mean - val)
    val_dev_mean = np.sqrt(np.sum(visit * val_dev, axis=-2))

    return val_mean_result, val_dev_mean


class BatchBufferFast(object):
    def __init__(self, threshold=15, size=20):
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, items):
        for item in items:
            if self.queue.qsize() <= self.threshold:
                self.queue.put(item)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()

        return None

    def get_len(self):
        return self.queue.qsize()

@ray.remote(num_gpus=0.08)
class BatchWorker:
    """
    Class which run in a dedicated thread to calculate the mini-batch for training
    """

    def __init__(self, rank, initial_checkpoint, batch_buffer, replay_buffer, shared_storage, config):
        self.rank = rank
        self.config = config
        self.batch_buffer = batch_buffer
        self.replay_buffer = replay_buffer
        self.shared_storage = shared_storage
        self.buffer = []
        self.model = MLPModel(config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["target_weights"]))
        self.model.to(torch.device("cuda"))
        self.model.eval()
        self.weight_step = 0
        # Fix random generator seed
        numpy.random.seed(self.config.seed)

    def set_weights(self, weights, weight_step=0):
        print("Setting weight!", weight_step)
        self.model.set_weights(weights)
        self.weight_step = weight_step
        return self.weight_step

    def calculate_bootstrap(self, observations, masks):
        """
        Run several MCTS for the given observations.

        :param observations:
        :param masks:
        :return:
        """

        # print('BOOTSTRAP_CALC', observations.shape)
        root_values, root_distributions, root_actions = run_multi_gail_support(observations, self.model, self.config)
        values = np.array(root_values)
        actions = np.array(root_actions)
        policies = np.array(root_distributions).astype(np.float32)
        policies /= policies.sum(axis=-1, keepdims=True)

        # print("BS", actions.shape,  policies.shape)
        values = values * masks
        # [256, ], [256, N, ACTION_DIM], [256, N]
        return values, actions, policies

    def buffer_add_batch(self, batch):
        self.buffer.append(batch)
        if len(self.buffer) > 5:
            self.buffer = self.buffer[1:]

    def buffer_get_batch(self):
        if len(self.buffer) > 0:
            data = copy.deepcopy(self.buffer[0])
            self.buffer = self.buffer[1:]
            return data
        return None

    def spin(self):
        while ray.get(self.shared_storage.get_info.remote("num_played_games")) < self.config.num_workers:
            time.sleep(1)

        batches = []
        while True:
            batch = self.get_batch()
            self.batch_buffer.push([batch])

            target_step = ray.get(self.shared_storage.get_info.remote("target_step"))
            running_games = ray.get(self.shared_storage.get_info.remote("running_games"))

            while running_games:
                time.sleep(1)
                running_games = ray.get(self.shared_storage.get_info.remote("running_games"))

            if target_step > self.weight_step:
                target_weights = ray.get(self.shared_storage.get_info.remote("target_weights"))
                self.model.set_weights(target_weights)
                self.weight_step = target_step


    def override_game_reward(self, observations, actions):
        observations = torch_utils.numpy_to_tensor(observations)
        actions = torch_utils.numpy_to_tensor(actions)
        return torch_utils.tensor_to_numpy(self.model.gail_reward(observations, actions))

    def get_batch(self):
        """
        :return:
        """
        (
            index_batch,
            observation_batch,
            next_observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            target_mu_batch,
            target_std_batch,
            gradient_scale_batch,
            raw_actions_batch,
            raw_policies_batch,
            mask_batch
        ) = ([], [], [], [], [], [], [], [], [], [], [], [])

        weight_batch = [] if self.config.PER else None

        n_total_samples, game_samples = ray.get([
            self.replay_buffer.get_n_total_samples.remote(),
            self.replay_buffer.sample_n_sqil_games.remote(self.config.batch_size // 2, self.config.batch_size // 2)]
        )

        """ 
            Reanalyzed
        """
        all_bootstrap_values = []
        x = time.time()

        # We need to calculate the bootstrap value.
        for i in range(self.config.num_unroll_steps_reanalyze + 1):
            begin_observation = []
            bootstrap_observation = []
            bootstrap_mask = []

            for (game_id, game_history, game_prob, game_pos, pos_prob) in game_samples:
                begin_index = self.config.stacked_observations - 1 + i
                bootstrap_index = self.config.stacked_observations - 1 + i + self.config.td_steps
                begin_observation.append(game_history.get_stacked_observations(
                        begin_index, self.config.stacked_observations)
                )

                if bootstrap_index > len(game_history.observation_history):
                    bootstrap_mask.append(0)
                    bootstrap_observation.append(
                        game_history.get_stacked_observations(
                        0, self.config.stacked_observations)
                    )

                else:
                    bootstrap_mask.append(1)
                    bootstrap_observation.append(
                        game_history.get_stacked_observations(
                            bootstrap_index, self.config.stacked_observations
                        )
                    )

            bootstrap_mask = np.array(bootstrap_mask)
            bootstrap_observation = np.array(bootstrap_observation)

            x = time.time()
            bootstrap_values, _, _ = self.calculate_bootstrap(bootstrap_observation, bootstrap_mask)
            bootstrap_values = bootstrap_values.reshape(-1, 1)

            # Reanalyze result.
            all_bootstrap_values.append(bootstrap_values)

            begin_observation = np.array(begin_observation)
            begin_mask = np.ones(begin_observation.shape[0])

            _, begin_actions, begin_policies = self.calculate_bootstrap(begin_observation, begin_mask)

            raw_actions_batch.append(begin_actions)
            raw_policies_batch.append(begin_policies)
            policy_mu, policy_std = calculate_mean_std(begin_actions, begin_policies)
            target_mu_batch.append(policy_mu)
            target_std_batch.append(policy_std)

        all_bootstrap_values = np.concatenate(all_bootstrap_values, axis=1)

        """
            Compute the targets.
        """
        for idx, (game_id, game_history, game_prob, game_pos, pos_prob) in enumerate(game_samples):
            values, actions, next_observations, masks = self.make_target(
                game_history, self.config.stacked_observations - 1, all_bootstrap_values[idx]
            )

            index_batch.append([game_id, game_pos])

            observation_batch.append(
                game_history.get_stacked_observations(
                    self.config.stacked_observations - 1, self.config.stacked_observations
                )
            )

            next_observation_batch.append(next_observations)
            action_batch.append(actions)
            value_batch.append(values)
            mask_batch.append(masks)

            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - (self.config.stacked_observations - 1),
                    )
                ]
                * len(actions)
            )

            if self.config.PER:
                weight_batch.append(1 / n_total_samples * game_prob * pos_prob)

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        observation_batch = np.array(observation_batch)
        next_observation_batch = np.array(next_observation_batch)
        mask_batch = np.array(mask_batch)
        action_batch = np.array(action_batch)
        value_batch = np.array(value_batch, dtype=np.float32)
        value_batch = value_batch.reshape(value_batch.shape[0], -1)
        # reward_batch = np.array(reward_batch)
        target_mu_batch = np.array(target_mu_batch)
        target_std_batch = np.array(target_std_batch)
        weight_batch = np.array(weight_batch)
        gradient_scale_batch = np.array(gradient_scale_batch)
        raw_actions_batch = np.array(raw_actions_batch)
        raw_policies_batch = np.array(raw_policies_batch)

        #   print('VS', value_batch.shape)
        #   print('TOT', time.time() - x)
        return (
            index_batch,
            self.weight_step,
            (
                observation_batch,          # [B, O_DIM]             * s0
                next_observation_batch,     # [B, STEP + TD_STEP + 1, O_DIM]   * s0,    s1,     s2,     ..., s_unroll
                action_batch,               # [B, STEP + TD_STEP + 1, A_DIM]   * X ,    a0,     a1,     ..., a_unroll-1
                value_batch,                # [B, STEP + 1]
                # reward_batch,              # [B, STEP + 1]          * X,     r0,     r1,     ...,
                target_mu_batch,            # [B, STEP + 1, A_DIM]   mu_0,  mu_1,   mu_2,   ...,
                target_std_batch,           # [B, STEP + 1, A_DIM]   std_0, std_1,  std_2,  ...,
                weight_batch,               # ...
                gradient_scale_batch,       # ...
                mask_batch,                  # [B, STEP + 1]
                raw_actions_batch,
                raw_policies_batch
            )
        )

    def compute_target_value(self, game_history, index=0, bootstrap_value=None):
        bootstrap_index = index + self.config.td_steps
        if bootstrap_value is not None:
            value = bootstrap_value * self.config.discount ** self.config.td_steps

        else:
            if bootstrap_index < len(game_history.root_values):
                root_values = (
                    game_history.root_values
                    if game_history.reanalysed_predicted_root_values is None
                    else game_history.reanalysed_predicted_root_values
                )

                last_step_value = (
                    root_values[bootstrap_index]
                )
                value = last_step_value * self.config.discount ** self.config.td_steps
            else:
                value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1: bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (reward) * self.config.discount ** i

        return value

    def make_target(self, game_history, state_index=0, bootstrap_value=None):
        """
        Generate targets for every unroll steps.
        """
        target_values  = []
        target_rewards = []
        actions = []
        target_masks = []
        target_next_observations = []
        """
            Target actions,     Target policies
            -----------------------------------
            action_vec_0        scalar_prob_0
            action_vec_1        scalar_prob_1
            action_vec_2        scalar_prob_2
            ...                 ...
        """

        """
            Obs_cur
            target_action_cur-1, target_action_cur,
        """

        # [UNROLL, NUM_ACTION]
        # [UNROLL, NUM_ACTION, ACTION_DIM]

        for current_index in range(
                state_index, state_index + self.config.num_unroll_steps + self.config.td_steps + 1
        ):
            if current_index - state_index <= self.config.num_unroll_steps_reanalyze:
                calculate_value = True
                value = bootstrap_value[current_index - state_index] * (self.config.discount ** self.config.td_steps)

            else:
                calculate_value = False

            if current_index < len(game_history.observation_history) - 1:
                if calculate_value:
                    target_values.append(value)

                target_next_observations.append(
                    game_history.get_stacked_observations(
                        current_index,
                        self.config.stacked_observations)
                )

                target_masks.append(1.)
                actions.append(game_history.action_history[current_index])

            elif current_index == len(game_history.observation_history) - 1:
                if calculate_value:
                    target_values.append(0.)

                target_next_observations.append(
                    game_history.get_stacked_observations(
                        0, self.config.stacked_observations)
                )

                actions.append(game_history.action_history[current_index])
                target_masks.append(0.)

            else:
                if calculate_value:
                    target_values.append(0.)

                target_next_observations.append(
                    game_history.get_stacked_observations(
                        0, self.config.stacked_observations)
                )

                # Magical number...
                random_actions = self.config.sample_random_actions(8)
                actions.append(random_actions[0])
                target_masks.append(0.)

        target_values = np.array(target_values).reshape(-1)
        target_masks = np.array(target_masks)
        target_next_observations = np.array(target_next_observations)
        actions = np.array(actions)

        return target_values, actions, target_next_observations, target_masks
