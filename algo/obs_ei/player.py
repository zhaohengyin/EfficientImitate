import ray
import numpy as np
import time
import torch
from algo.obs_ei.mcts import MCTS
from algo.obs_ei.model import MuZeroResidualNetwork
from mcts_tree_sample.mcts import run_multi_gail_support


class GameHistory:
    """
    Store only useful information of a self-play game.
    """
    def __init__(self):
        self.observation_history = []
        self.render_history = []
        self.action_history = []
        self.reward_history = []
        self.reward_true_history = []
        self.frames = []
        self.trees = []
        self.phys_states = []

        # self.to_play_history = []

        self.child_visits = []
        self.child_values = []
        self.child_qinits = []
        self.root_values = []
        self.root_actions = []

        self.reanalysed_predicted_root_values = None

        # For PER
        self.priorities = None
        self.game_priority = None

    def subset(self, pos, duration):
        if pos < 0:
            pos = 0

        res = GameHistory()
        res.observation_history = self.observation_history[pos:pos + duration]
        res.action_history = self.action_history[pos:pos + duration]
        res.reward_history = self.reward_history[pos:pos + duration]
        res.reward_true_history = self.reward_true_history[pos:pos + duration]
        res.child_visits = self.child_visits[pos:pos + duration]
        res.root_values = self.root_values[pos:pos + duration]
        res.root_actions = self.root_actions[pos:pos + duration]

        if self.reanalysed_predicted_root_values is not None:
            res.reanalysed_predicted_root_values = self.reanalysed_predicted_root_values[pos:pos + duration]

        if self.priorities is not None:
            res.priorities = self.priorities[pos:pos + duration]

        return res

    def store_search_statistics(self, root):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            root_action = []
            root_child_qinit = []
            root_child_value = []
            root_child_visit = []
            for action_id in root.action_ids:
                root_child_visit.append(root.children[action_id].visit_count / sum_visits)
                root_action.append(root.actions[action_id])
                root_child_qinit.append(root.children[action_id].q_init)
                root_child_value.append(root.children[action_id].value())

            self.child_qinits.append(root_child_qinit)
            self.child_values.append(root_child_value)
            self.child_visits.append(root_child_visit)
            self.root_actions.append(root_action)
            self.root_values.append(root.value())
            # print('STORE', root_child_visit, root_action, root.value())
        else:
            self.root_values.append(None)

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


class Player:
    def __init__(self, config, seed, n_parallel):
        self.model = MuZeroResidualNetwork(config)
        self.model.to(torch.device("cuda"))
        self.model.eval()

        self.eval_model = MuZeroResidualNetwork(config)
        self.eval_model.to(torch.device("cuda"))
        self.eval_model.eval()

        self.envs = [config.new_game(seed+i) for i in range(n_parallel)]
        self.eval_envs = [config.new_game(1 + i) for i in range(10)]

        self.n_parallel = n_parallel
        self.config = config

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def set_eval_weights(self, weights):
        self.eval_model.set_weights(weights)

    def run_eval(self, trained_steps=0):
        observations = [env.reset() for env in self.eval_envs]

        game_histories = [GameHistory() for i in range(10)]

        for i, game_history in enumerate(game_histories):
            game_history.action_history.append(self.config.sample_random_actions(1).reshape(-1))
            game_history.observation_history.append(observations[i])
            game_history.reward_history.append(0)
            game_history.reward_true_history.append(0)

        forward_observations = [game_history.get_stacked_observations(-1,
                                                                      self.config.stacked_observations, ) for
                                game_history in game_histories]
        dones = [0 for i in range(10)]

        steps = 0
        while sum(dones) < 10:
            steps += 1
            forward_observations = np.array(forward_observations)
            root_values, root_distributions, root_actions = run_multi_gail_support(forward_observations, self.model,
                                                                                   self.config)
            root_values = np.array(root_values)  # [EsNV_ID, 1]
            root_actions = np.array(root_actions)  # [ENV_ID, N_ACTIONS, ACTION_DIM]
            root_visit_counts = np.array(root_distributions).astype(np.float32)  # [ENV_ID, N_ACTIONS, 1]

            if steps % 10 == 0:
                print(root_values.shape, root_actions.shape, root_visit_counts.shape, root_visit_counts)
            next_observations = []

            for i, game_history in enumerate(game_histories):
                if dones[i]:
                    next_observations.append(observations[i])
                    continue

                mcts_value = root_values[i]
                mcts_action = root_actions[i]
                mcts_visit_count = root_visit_counts[i]

                action = self.select_action(mcts_action, mcts_visit_count, 0)
                next_observation, reward, done, info = self.eval_envs[i].step(action)
                next_observations.append(next_observation)

                if done:
                    dones[i] = 1

                game_history.action_history.append(action)
                game_history.observation_history.append(next_observation)
                game_history.reward_history.append(0)
                game_history.reward_true_history.append(reward)
                game_history.child_visits.append(mcts_visit_count)
                game_history.root_actions.append(mcts_action)
                game_history.root_values.append(mcts_value)

            forward_observations = [game_history.get_stacked_observations(-1,
                                                                          self.config.stacked_observations, ) for
                                    game_history in game_histories]

        eval_reward = 0

        for game_history in game_histories:
            print("Eval: Len={}, Reward={}".format(len(game_history.reward_true_history),
                                                   sum(game_history.reward_true_history)))
            eval_reward += sum(game_history.reward_true_history)

        eval_avg_reward = eval_reward / len(game_histories)
        print("Eval = ", eval_reward / len(game_histories))
        return game_histories, eval_avg_reward

    def run(self, trained_steps=0):
        observations = [env.reset() for env in self.envs]

        game_histories = [GameHistory() for i in range(self.n_parallel)]

        for i, game_history in enumerate(game_histories):
            game_history.action_history.append(self.config.sample_random_actions(1).reshape(-1))
            game_history.observation_history.append(observations[i])
            game_history.reward_history.append(0)
            game_history.reward_true_history.append(0)

        forward_observations = [game_history.get_stacked_observations(-1,
                    self.config.stacked_observations,) for game_history in game_histories]
        # observations = np.array(observations)
        dones = [0 for i in range(self.n_parallel)]

        steps = 0
        while sum(dones) < self.n_parallel:
            steps += 1
            forward_observations = np.array(forward_observations)
            root_values, root_distributions, root_actions = run_multi_gail_support(forward_observations, self.model,
                                                                                   self.config)
            root_values = np.array(root_values)  # [EsNV_ID, 1]
            root_actions = np.array(root_actions)  # [ENV_ID, N_ACTIONS, ACTION_DIM]
            root_visit_counts = np.array(root_distributions).astype(np.float32)  # [ENV_ID, N_ACTIONS, 1]

            if steps % 10 == 0:
                print(root_values.shape, root_actions.shape, root_visit_counts.shape, root_visit_counts)

            for i, game_history in enumerate(game_histories):
                if dones[i]:
                    continue

                mcts_value = root_values[i]
                mcts_action = root_actions[i]
                mcts_visit_count = root_visit_counts[i]

                action = self.select_action(mcts_action, mcts_visit_count,
                                            self.config.visit_softmax_temperature_fn(trained_steps))
                next_observation, reward, done, info = self.envs[i].step(action)

                if done:
                    dones[i] = 1

                game_history.action_history.append(action)
                game_history.observation_history.append(next_observation)
                game_history.reward_history.append(0)
                game_history.reward_true_history.append(reward)
                game_history.child_visits.append(mcts_visit_count)
                game_history.root_actions.append(mcts_action)
                game_history.root_values.append(mcts_value)

            forward_observations = [game_history.get_stacked_observations(-1,
                                                                          self.config.stacked_observations, ) for
                                    game_history in game_histories]

        for game_history in game_histories:
            print("Len={}, Reward={}".format(len(game_history.reward_true_history),
                                             sum(game_history.reward_true_history)))
        return game_histories

    def select_action(self, actions, visit_counts, temperature):
        """
            Select action according to the visit count distribution and the temperature.
            The temperature is changed dynamically with the visit_softmax_temperature function
            in the config.
		"""

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]

        elif temperature == float("inf"):
            action_id = np.random.choice(visit_counts.shape[0])
            action = actions[action_id]

        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action_id = np.random.choice(visit_counts.shape[0], p=visit_count_distribution)
            action = actions[action_id]

        return action


def reward_shaping_func(r, delta=1, amp=1):
    if delta == 0:
        return r
    return (r // delta) * delta * amp


# Deprecated.
@ray.remote(num_gpus=0.06)
class PlayerWorker:
    """
    Self-play workers running in parallel.

    """
    def __init__(self, config, model, seed):
        self.config = config
        self.game = config.new_game(seed)
        self.model = model
        self.seed = seed

    def spin(self, shared_storage, replay_buffer, test_mode=False):
        """

        :param shared_storage:
        :param replay_buffer:
        :param test_mode:
        :return:
        """
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):

            self.model.set_weights.remote(ray.get(shared_storage.get_info.remote("weights")))

            self.model.lock.remote()
            if ray.get(shared_storage.get_info.remote("force_selfplay_halt")):
                time.sleep(1000)
                continue

            shared_storage.inc_counter.remote("rolled_games")

            if not test_mode:
                # Keep playing new games.
                game_history = self.play_game(
                    temperature=self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    temperature_threshold=self.config.temperature_threshold,
                    render=False,
                )

                print("Saving game to replay buffer!!!!")
                print("Total reward:= {}, True reward:= {}".format(
                    sum(game_history.reward_history), sum(game_history.reward_true_history)
                ))

                replay_buffer.save_game.remote(game_history, shared_storage)
                # replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    temperature=0,
                    temperature_threshold=self.config.temperature_threshold,
                    render=False
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": np.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )

            self.model.unlock.remote()
            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)

            # Waiting for the training to finish.
            if not test_mode and self.config.ratio_lower_bound:

                while True:
                    training_step = ray.get(shared_storage.get_info.remote("training_step"))
                    desired_games = (1 + training_step // (self.config.epoch_repeat *
                                                           self.config.num_workers *
                                                           self.config.max_moves)) * self.config.num_workers

                    # Note that there may exist other self-playing games that are still running.
                    if ray.get(shared_storage.get_info.remote("rolled_games")) >= desired_games \
                        and ray.get(shared_storage.get_info.remote("training_step")) < self.config.training_steps \
                        and not ray.get(shared_storage.get_info.remote("terminate")):
                        time.sleep(1)
                    else:
                        break


        self.close_game()

    def test(self, shared_storage):
        self.model.set_weights.remote(ray.get(shared_storage.get_info.remote("weights")))
        game_history = self.play_game(
            temperature=0,
            temperature_threshold=self.config.temperature_threshold,
            render=False,
            add_exploration_noise=False,
            record=True
        )

        return game_history

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        n_child_action = len(node.children)

        visit_counts = np.array(
            [node.children[action_id].visit_count for action_id in node.action_ids], dtype="int32"
        )

        if temperature == 0:
            action = node.actions[node.action_ids[np.argmax(visit_counts)]]

        elif temperature == float("inf"):
            action_id = np.random.choice(n_child_action)
            action = node.actions[action_id]

        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action_id = np.random.choice(n_child_action, p=visit_count_distribution)
            action = node.actions[action_id]

        return action

    def close_game(self):
        self.game.close()

    def play_game(
            self, temperature, temperature_threshold, render, add_exploration_noise=True, record=False
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """

        observation = self.game.reset()

        game_history = GameHistory()
        game_history.action_history.append(self.config.sample_random_actions(1).reshape(-1))
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.reward_true_history.append(0)

        """
            Obs_history:    [Obs0,  Obs1,  Obs2, ...]
            Act_history:    [0,     Act0,  Act1, ...]
            Rew_history:    [0,     Rew0,  Act1, ...]
            Root_history:   [Root0, Root1, Root2, ...]
        """

        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            while (
                    not done and len(game_history.action_history) <= self.config.max_moves
            ):

                stacked_observations = game_history.get_stacked_observations(
                    -1,
                    self.config.stacked_observations,
                )

                # print('Stacked observation', stacked_observations.shape)
                # Choose the action
                import time

                # print('SHAPE', stacked_observations.shape)
                root = MCTS(self.config).run(
                    model=self.model,
                    observation=stacked_observations.reshape(1, *stacked_observations.shape),
                    add_exploration_noise=add_exploration_noise
                )

                action = self.select_action(
                    root,
                    temperature
                    if not temperature_threshold
                       or len(game_history.action_history) < temperature_threshold
                    else 0,
                )

                if record:
                    print('Now step', len(game_history.frames))
                    frame = self.game.env.render(mode='rgb_array',
                                                 height=100,
                                                 width=100,
                                                 camera_id=0)
                    game_history.frames.append(frame)
                    game_history.phys_states.append(self.game.env.get_phy_state())
                    game_history.trees.append(root.dump_tree())

                # print('ACTION', action)
                observation, reward, done, _ = self.game.step(action)
                reward_clipped = 0 # reward_shaping_func(reward, self.config.reward_delta, self.config.reward_amp)

                game_history.store_search_statistics(root)
                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_true_history.append(reward)
                game_history.reward_history.append(reward_clipped)

        return game_history

