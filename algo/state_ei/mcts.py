import math
import time
import numpy as np
import torch_utils
from torch_utils import *
import ray
import torch
import torch.nn.functional as F

class Node:
    """
    Node class in the search tree.
    """

    def __init__(self, reward=0, prior=0.):
        self.action_ids = []
        self.actions = {}
        self.children = {}
        self.q_init = None
        self.visit_count = 0
        self.value_sum = 0
        self.reward = reward
        self.hidden_state = None
        self.prior = prior
        self.last_ucb_prior_score = 0
        self.last_ucb_value_score = 0
        self.last_ucb_score = 0

        pass

    def value(self):
        """
        The estimated value of a given node, which is updated during backpropagation.
        :return: float value
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def dump_tree(self):
        tree_info = {'act': [], 'c': [], 'vis': [], 'uv': [], 'up': [], 'us': [], 'q': []}

        for aid in self.action_ids:
            action = self.actions[aid]
            child = self.children[aid]

            child_info = child.dump_tree()
            tree_info['act'].append(action)
            tree_info['c'].append(child_info)
            tree_info['r'] = self.reward
            tree_info['vis'].append(child.visit_count)
            tree_info['q'].append(child.q_init)
            # tree_info['val'].append(child.value())
            tree_info['uv'].append(child.last_ucb_value_score)
            tree_info['us'].append(child.last_ucb_score)
            tree_info['up'].append(child.last_ucb_prior_score)

        return tree_info

    def expanded(self):
        """
        Whether the node has been expanded.
        :return: Boolean
        """

        return len(self.children) > 0

    def get_mean_q(self, is_root, parent_q, discount):
        total_unsigned_q = 0.0
        total_visits = 0

        for aid in self.action_ids:
            child = self.children[aid]

            if child.visit_count > 0:
                qsa = child.reward + discount * child.value()
                total_unsigned_q += qsa
                total_visits += 1

        if is_root and total_visits > 0:
            mean_q = total_unsigned_q / total_visits
        else:
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1)

        return mean_q

    def expand(self, reward, actions, hidden_state, child_q=None):
        """
        Expand a given (leaf) node.
        :param reward:          The estimated reward when transit into this state in MCTS.
        :param actions:         The sampled legal actions starting from this state.
        :param hidden_state:    The state representation of this state.
        :return: void
        """

        for i, act in enumerate(actions):
            self.action_ids.append(i)
            self.actions[i] = act

        self.reward = reward
        self.hidden_state = hidden_state

        # Add children. Note that their prior are the same.
        uniform_prior = 1 / len(actions)
        for i, id in enumerate(self.action_ids):
            self.children[id] = Node(prior=uniform_prior)

            if child_q is not None:
                self.children[id].q_init = child_q[i]

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """

        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MCTS:
    """
    Monte Carlo Tree Search for continuous space, which is based on action sampling.
    """
    def __init__(self, config):
        self.config = config

    def ucb_score(self, parent, child, min_max_stats, parent_mean_q):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )

        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = child.reward + self.config.discount * child.value()

        else:
            if child.q_init is not None:
                value_score = child.q_init
            else:
                value_score = parent_mean_q

        value_score = min_max_stats.normalize(value_score)
        child.last_ucb_prior_score = prior_score
        child.last_ucb_value_score = value_score
        child.last_ucb_score = prior_score + value_score

        return prior_score + value_score

    def run(self,
            model,
            observation,
            add_exploration_noise=False):
        """
        The search function.

        :param model: The neural network model to calculate dynamics, reward, value, policy.
        :param observation: The current observation of the environment.
        :param add_exploration_noise:
        :return: root of the tree.

        """
        root = Node(prior=0)

        # Calculate the root information
        (
            root_predicted_value,
            root_reward,
            policy_info,
            hidden_state,
        ) = ray.get(model.initial_inference.remote(torch_utils.numpy_to_tensor(observation)))

        # Compute scalar reward.
        if root_reward.size(-1) != 1:
            root_reward = support_to_scalar(root_reward, self.config.reward_support_size).item()
        else:
            root_reward = root_reward.squeeze().item()

        sampled_actions = ray.get(model.sample_mixed_actions.remote(policy_info, self.config, True))[0]
        # print(sampled_actions.shape)
        # print('Sampled Actions', sampled_actions)
        min_max_stats = MinMaxStats()

        child_q = ray.get(model.eval_q.remote(torch_utils.numpy_to_tensor(observation),
                                              torch_utils.numpy_to_tensor(sampled_actions)))
        child_q = child_q.tolist()

        for q in child_q:
            min_max_stats.update(q)

        root.expand(reward=root_reward, actions=sampled_actions, hidden_state=hidden_state, child_q=child_q)

        # Add exploration noise.
        if add_exploration_noise:
            root.add_exploration_noise(dirichlet_alpha=self.config.root_dirichlet_alpha,
                                       exploration_fraction=self.config.root_exploration_fraction)

        # Simulation
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            is_root = True
            parent_q = 0

            while node.expanded():
                # Keep selecting
                mean_q = node.get_mean_q(is_root, parent_q, self.config.discount)
                parent_q = mean_q
                is_root = False

                action_id, next_node = self.select_child(node, min_max_stats, mean_q)
                action = node.actions[action_id]

                node = next_node
                search_path.append(node)

            # When we exit the loop, a leaf node is reached.
            leaf_parent = search_path[-2]
            (
                leaf_value, leaf_reward,
                leaf_policy, leaf_hidden_state,
            ) = ray.get(model.recurrent_inference.remote(leaf_parent.hidden_state,
                                                         torch_utils.numpy_to_tensor(action.reshape(1, *action.shape))))

            if leaf_value.size(-1) != 1:
                leaf_value = support_to_scalar(leaf_value, self.config.support_size).item()
            else:
                leaf_value = leaf_value.squeeze().item()

            if leaf_reward.size(-1) != 1:
                leaf_reward = support_to_scalar(leaf_reward, self.config.reward_support_size).item()

            else:
                # Turn the logits into adversarial rewards.
                leaf_reward = -torch.log(1 - F.sigmoid(leaf_reward) + 1e-6)
                leaf_reward = leaf_reward.squeeze().item()

            leaf_actions_policy = ray.get(model.sample_mixed_actions.remote(leaf_policy, self.config, False))[0]

            # leaf_actions_policy = ray.get(model.sample_actions.remote(
            #     leaf_policy, self.config.mcts_num_policy_samples + self.config.mcts_num_random_samples)
            # )[0]
            # leaf_actions_random = self.config.sample_random_actions(self.config.mcts_num_random_samples)

            # leaf_actions = np.array([[-0.80], [-0.40], [-0.2],  [0.00], [0.2], [0.40], [0.80]]).astype(np.float32)

            leaf_actions = leaf_actions_policy # np.concatenate([leaf_actions_policy, leaf_actions_random], axis=0)
            # print('Sampled Actions', leaf_actions)

            node.expand(
                leaf_reward,
                leaf_actions,
                leaf_hidden_state,
            )

            self.backpropagate(search_path, leaf_value, min_max_stats)

        return root


    def select_child(self, node, min_max_stats, mean_q):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats, mean_q)
            for action, child in node.children.items()
        )
        action = np.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats, mean_q) == max_ucb
            ]
        )
        return action, node.children[action]

    def backpropagate(self, search_path, value, min_max_stats):
        """
        There is only one player in our experiments.
        """

        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.config.discount * node.value())
            value = node.reward + self.config.discount * value



class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / max([(self.maximum - self.minimum), 0.01])
        return value
