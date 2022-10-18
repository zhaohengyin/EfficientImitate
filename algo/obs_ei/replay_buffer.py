import copy
import time

import numpy
import ray
import torch

import pickle_utils
import numpy as np


class ExpertGameHistory:
    """
    Store only useful information of a self-play game.
    """
    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.reanalysed_predicted_root_values = None

    def subset(self, pos, duration):
        if pos < 0:
            pos = 0

        res = ExpertGameHistory()
        res.observation_history = self.observation_history[pos:pos + duration]
        res.action_history = self.action_history[pos:pos + duration]
        res.reward_history = self.reward_history[pos:pos + duration]

        return res

    def get_stacked_observations(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """

        # Convert to positive index
        index = index % len(self.observation_history)

        # [t, t-1, t-2, ...]
        # stacked_observations = self.observation_history[index].copy()
        observations = []

        for past_observation_index in reversed(
            range(index + 1 - num_stacked_observations, index + 1)
        ):

            if 0 <= past_observation_index:
                observations.append(self.observation_history[past_observation_index])

            else:
                observations.append(self.observation_history[0])

        stacked_observations = np.concatenate(observations, axis=0)
        return stacked_observations


def get_expert_buffer(path):
    buffer = {}
    expert_trajectories = pickle_utils.load_data(path)
    for idx, traj in enumerate(expert_trajectories):
        game_history = ExpertGameHistory()
        print('Expert Trajectory Length =', len(traj['image']))
        game_history.observation_history = traj['image']
        game_history.action_history = [traj['act'][0]] + traj['act']  # we need a padding at the front.
        game_history.reward_history = [1.0 for _ in range(len(game_history.action_history))]

        index = - idx - 1   # For convenience, we use the -1, -2, ..., -n to index the expert demos.
        buffer[index] = game_history

    return buffer


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config, test_throughput=False):
        self.config = config

        self.buffer = copy.deepcopy(initial_buffer)
        self.expert_buffer = get_expert_buffer(config.expert_demo_path)

        def extend(buffer):
            new_buffer = {}
            counter = 0
            for k, v in buffer.items():
                for i in range(1):
                    new_buffer[counter] = v
                    counter += 1
            return new_buffer

        if test_throughput:
            self.buffer = extend(self.buffer)

        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )

        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        self.last_game_reward = []
        self.last_game_true_reward = []

    def sample_representation_batch(self, batchsize):
        selected_games = numpy.random.choice(list(self.expert_buffer.keys()), batchsize)
        ret = []

        for game_id in selected_games:
            game = self.expert_buffer[game_id]
            game_pos = numpy.random.choice(len(game.observation_history) - 16)
            ret.append(game.subset(max([0, game_pos - self.config.stacked_observations + 1]), 16))

        batch_obs = []
        batch_act = []
        batch_mask = []
        for game in ret:
            obs = []
            act = []
            mask = []

            current_index = 0
            for i in range(self.config.num_unroll_steps_reanalyze + 1):
                obs.append(game.get_stacked_observations(i, self.config.stacked_observations))

                if i >= len(game.observation_history):
                    act.append(game.action_history[current_index])
                    mask.append(1.0)
                else:
                    act.append(game.action_history[i])
                    mask.append(0.0)
                    current_index = i

            obs = np.array(obs) # [NUM_STEP, C, H, W】
            act = np.array(act) # [NUM_STEP, A]
            mask = np.array(mask).reshape(-1, 1) # [NUM_STEP, 1]

            batch_mask.append(mask)
            batch_obs.append(obs)
            batch_act.append(act)

        return np.array(batch_obs), np.array(batch_act), np.array(batch_mask)


    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
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
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (reward) * self.config.discount ** i

        return value

    def save_game(self, game_history, shared_storage=None):
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(game_history, i)
                        )
                        ** self.config.PER_alpha
                    )
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
                game_history.game_priority = numpy.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.num_played_games % 4 == 0:
            debug_games = {}
            for k in list(self.buffer.keys())[::-1]:
                debug_games[k] = self.buffer[k]
                if len(debug_games) > 4:
                    break
            import os
            pickle_utils.save_data(debug_games, os.path.join(self.config.results_path,
                                                             'dbg_{}.pkl'.format(self.num_played_games)))


        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        self.last_game_reward.append(sum(game_history.reward_history))
        self.last_game_true_reward.append(sum(game_history.reward_true_history))

        # save_data(self.buffer, './replay.pkl')

        if shared_storage:
            # print("Replay buffer update storage")
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)
            # shared_storage.set_info.remote("num_total_samples", self.get_n_total_samples())

            # Calculate mean training reward in the last.
            num_last_game = min([len(self.last_game_reward), 8])
            if num_last_game > 0:
                last_mean_reward = sum(self.last_game_reward[-num_last_game:]) / num_last_game
                last_mean_true_reward = sum(self.last_game_true_reward[-num_last_game:]) / num_last_game
                print("MEAN TRUE:", last_mean_true_reward)
                shared_storage.set_info.remote("mean_training_reward", last_mean_reward)
                shared_storage.set_info.remote("mean_training_true_reward", last_mean_true_reward)

    def sample_position(self, game_history, force_uniform=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            # In the target value computation, we avoid going out of the rollout.
            position_index = numpy.random.choice(len(game_history.root_values) - 16)
            position_prob = 1 / (len(game_history.root_values) - self.config.td_steps)
        return position_index, position_prob

    def get_buffer(self):
        return self.buffer

    def get_n_total_samples(self):
        return self.total_samples

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = numpy.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= numpy.sum(game_probs)
            game_index = numpy.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = numpy.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_sqil_games(self, n_games, n_expert_games):
        return self.sample_n_games(n_games) + self.sample_n_expert_games(n_expert_games)

    def sample_n_expert_games(self, n_games):
        # Without any priority
        selected_games = numpy.random.choice(list(self.expert_buffer.keys()), n_games)
        ret = []

        for game_id in selected_games:
            game = self.expert_buffer[game_id]
            game_pos = numpy.random.choice(len(game.observation_history) - 12)
            pos_prob = 1 / len(game.observation_history)

            ret.append((game_id, game.subset(max([0, game_pos - self.config.stacked_observations + 1]), 12),
                        1, game_pos, pos_prob))

        return ret

    def sample_n_games(self, n_games, force_uniform=False):
        # game_id, game_history, game_prob, game_pos, pos_prob
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)

            game_probs = numpy.array(game_probs, dtype="float32")
            game_probs /= numpy.sum(game_probs)

            game_prob_dict = dict([(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)])
            selected_games = numpy.random.choice(game_id_list, n_games, p=game_probs)

        else:
            selected_games = numpy.random.choice(list(self.buffer.keys()), n_games)
            game_prob_dict = {}

        ret = []
        for game_id in selected_games:
            game = self.buffer[game_id]
            game_pos, pos_prob = self.sample_position(game, True)
            ret.append((game_id, game.subset(max([0, game_pos - self.config.stacked_observations + 1]), 12),
                                             game_prob_dict.get(game_id), game_pos, pos_prob))

        return ret

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_game_history_reanalyze(self, game_ids, game_history_reanalyze_values):
        # The element could have been removed since its selection and update

        cmp_id = next(iter(self.buffer))
        for i, game_id in enumerate(game_ids):
            if cmp_id <= game_id:
                self.buffer[game_id].reanalysed_predicted_root_values = game_history_reanalyze_values[i]

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            if game_id < 0:
                # This is an expert demo.
                continue

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update game priorities
                self.buffer[game_id].game_priority = numpy.max(
                    self.buffer[game_id].priorities
                )

