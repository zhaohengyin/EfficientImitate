import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_utils
from torch_utils import *
from abc import ABC, abstractmethod


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


'''
    Observational stuffs....
'''


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


class RepresentationNetwork(torch.nn.Module):
    """

    Example: (3, 48, 48) --> (32, 6, 6) --> 1024 --> 128

    """

    def __init__(
            self,
            observation_shape,
            stacked_observations,
            momentum=0.1,
            f_dim=128
    ):
        super().__init__()

        self.f_dim = f_dim
        self.input_dim = observation_shape[0] * stacked_observations
        self.cnn_net = nn.Sequential(nn.Conv2d(in_channels=observation_shape[0] * stacked_observations,
                                               out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=2),
                                     # nn.BatchNorm2d(32, momentum=momentum),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32,
                                               out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                     # nn.BatchNorm2d(64, momentum=momentum),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32,
                                               out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                     # nn.BatchNorm2d(64, momentum=momentum),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32,
                                               out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
        )

        self.cnn_output_dim = self.get_frontend_output_dim()
        self.mlp = nn.Sequential(nn.Linear(self.cnn_output_dim, self.f_dim),
                                 nn.ReLU())

    def forward(self, x):
        x = self.cnn_net(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x

    def get_frontend_output_dim(self):
        x = self.cnn_net(torch.randn(3, self.input_dim, 48, 48))
        x = x.reshape(3, -1)
        return x.size(1)

# If predicts next state.
class PredNetwork(nn.Module):
    def __init__(self, h_dim, a_dim, momentum=0.1):
        super(PredNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(h_dim + a_dim, 2 * h_dim),
                                     nn.ReLU(),
                                     nn.Linear(2 * h_dim, h_dim),
                                     nn.ReLU(),
                                     )

    def forward(self, x, a):
        x_ = torch.cat((x, a),dim=1)
        return self.network(x_)


class ActorCriticNetwork(nn.Module):
    def __init__(self, f_dim, out_dim, layers):
        super(ActorCriticNetwork, self).__init__()
        self.mlp = mlp(f_dim, layers, output_size=out_dim, use_bn=False)

    def forward(self, x):
        return self.mlp(x)


class RewardNetwork(nn.Module):
    def __init__(self, f_dim, a_dim, layers):
        super(RewardNetwork, self).__init__()
        self.mlp = mlp(f_dim + a_dim, layers, 1, use_bn=False)

    def forward(self, x, a):
        x = x.reshape(x.size(0), -1)
        return self.mlp(torch.cat((x, a), dim=-1))


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(self, config):
        super().__init__()
        observation_shape = config.observation_shape
        stacked_observations = config.stacked_observations
        action_space_size = config.action_space_size
        support_size = config.support_size

        bn_mt = config.bn_mt


        proj_hid = config.rep_hi #1024
        proj_out = config.rep_hi #1024
        pred_hid = config.rep_lo #512
        pred_out = config.rep_hi #1024

        init_zero = False
        state_norm = False

        self.config = config
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.state_norm = state_norm

        self.support_size = support_size
        self.support_step = config.support_step
        self.reward_support_size = config.reward_support_size
        self.reward_support_step = config.reward_support_step
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        self.full_reward_support_size = 2 * config.reward_support_size + 1

        self.gamma = config.discount

        self.f_dim = 128

        self.representation_network = RepresentationNetwork(
                observation_shape,
                stacked_observations,
                momentum=bn_mt,
                f_dim=self.f_dim
            )

        # Output =(64, 6, 6)

        self.dynamics_network = PredNetwork(self.f_dim, self.action_space_size, momentum=bn_mt)

        self.reward_network = RewardNetwork(f_dim=self.f_dim, a_dim=self.action_space_size,
                                            layers=self.config.mlp_reward)

        self.policy_network = ActorCriticNetwork(f_dim=self.f_dim,
                                                 out_dim=self.action_space_size * 2, layers=self.config.mlp_policy)

        self.policybc_network = ActorCriticNetwork(f_dim=self.f_dim,
                                                 out_dim=self.action_space_size * 2, layers=self.config.mlp_policy)

        self.value_network = ActorCriticNetwork(f_dim=self.f_dim,
                                                 out_dim=self.full_support_size, layers=self.config.mlp_value)


        self.projection_in_dim = self.f_dim
        self.projection = nn.Sequential(
            nn.Linear(self.projection_in_dim, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid, momentum=bn_mt),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid, momentum=bn_mt),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_out),
            nn.BatchNorm1d(self.proj_out)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid, momentum=bn_mt),
            nn.ReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

    def sample_actions(self, policy, n):
        action_dim = policy.shape[-1] // 2
        mu, log_std = policy[:, :action_dim], policy[:, action_dim:]
        sigma = torch.exp(log_std) + 5e-3

        distr = torch.distributions.Normal(mu, sigma)
        action = distr.sample(torch.Size([n]))
        action = torch.nn.functional.tanh(action)
        action = action.permute(1, 0, 2)

        return action

    def sample_mixed_actions(self, policy, config, is_root):
        n_batchsize = policy.shape[0]
        n_policy_action = config.mcts_num_policy_samples
        n_random_action = config.mcts_num_random_samples

        #if is_root:
            # RETURNING, [BATCHSIZE, N, ACTION_DIM]

        if config.explore_type == 'add':
            # random_actions = config.sample_random_actions_fast(n_random_action * n_batchsize)
            # random_actions = random_actions.reshape(n_batchsize, -1, config.action_space_size)

            pi_mu, pi_logstd, exp_mu, exp_logstd = torch.chunk(policy, chunks=4, dim=-1)

            pi_sigma = torch.exp(pi_logstd)
            pi_distr = SquashedNormal(pi_mu, pi_sigma)

            exp_sigma = torch.exp(exp_logstd)
            exp_distr = SquashedNormal(exp_mu, exp_sigma)

            policy_action = pi_distr.sample(torch.Size([n_policy_action]))  # [n_pol, batchsize, a_dim]
            policy_action = torch_utils.tensor_to_numpy(policy_action.permute(1, 0, 2))

            expert_action = exp_distr.sample(torch.Size([n_random_action]))
            expert_action = torch_utils.tensor_to_numpy(expert_action.permute(1, 0, 2))

            if n_random_action > 0:
                return np.concatenate([policy_action, expert_action], axis=1)
            else:
                return policy_action


        else:
            assert False, 'exploration type wrong! Get: {}'.format(config.explore_type)

    def prediction(self, encoded_state):
        encoded_state = encoded_state.reshape(encoded_state.size(0), -1)
        policy = self.policy_network(encoded_state)
        policybc = self.policybc_network(encoded_state)
        value = self.value_network(encoded_state)
        return torch.cat([policy, policybc], dim=-1), value

    def representation(self, observation):
        assert observation.max() < 10, "We assume that the observation is already pre-processed."
        encoded_state = self.representation_network(observation)
        return encoded_state

    def dynamics(self, encoded_state, action):
        """

               :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
               :param action: shape: [Batchsize, Action_Dim]
               :return:
        """
        # x = torch.cat((encoded_state, action), dim=-1)
        next_encoded_state = self.dynamics_network(encoded_state, action)
        reward = self.reward_network(encoded_state, action)
        return next_encoded_state, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)

        # reward equal to 0 for consistency
        reward = torch.zeros(observation.size(0), 1).to(observation.device)

        return value, reward, policy_logits, encoded_state,

    def recurrent_inference(self, encoded_state, action):
        """

        :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
        :param action: shape: [Batchsize, Action_Dim]
        :return:
        """
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state

    def project(self, hidden_state, with_grad=True):
        hidden_state = hidden_state.view(-1, self.projection_in_dim)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()

    def value(self, h):
        _, v = self.prediction(h)
        return v

    def policy(self, h):
        pi, _ = self.prediction(h)
        return pi

    def eval_q(self, obs, actions):
        # obs shape = [BATCHSIZE, C, H, W]
        if len(obs.shape) == 4:
            if len(actions.shape) == 2:
                actions = actions.reshape(1, *actions.shape)

            # Obs shape = [BATCHSIZE, C, H, W]
            # actions shape = [BATCHSIZE, N, A_DIM]

            batch_shape = obs.size(0)
            num_actions = actions.size(1)

            obs_expand = obs.reshape(
                obs.size(0), 1, obs.shape[1], obs.shape[2], obs.shape[3]
            ).repeat(1, actions.size(1), 1, 1, 1)

            obs_expand = obs_expand.reshape(
                obs_expand.size(0) * obs_expand.size(1), obs_expand.size(2), obs_expand.size(3), obs_expand.size(4)
            )
            actions = actions.reshape(actions.size(0) * actions.size(1), -1)

            h = self.representation(obs_expand)
            next_h, r = self.dynamics(h, actions)
            _, next_v = self.prediction(next_h)

            r = -torch.log(1 - F.sigmoid(r) + 1e-6)
            next_v = support_to_scalar(next_v, self.support_size, self.support_step)

            r = r.reshape(batch_shape, num_actions, 1)
            next_v = next_v.reshape(batch_shape, num_actions, 1)

            assert len(next_v.shape) == 3, 'Next v error'.format(next_v.shape)
            assert len(r.shape) == 3, 'R shape error:{}'.format(r.shape)
            values = r + self.gamma * next_v

            return values.squeeze()

        elif len(obs.shape) == 3:
            # Obs shape = [C, W, H]
            # Obs shape = [N, A_DIM]
            obs_expand = obs.reshape(1, obs.size(0), obs.size(1), obs.size(2)).repeat(actions.size(0), 1, 1, 1)
            # [N, C, W, H]

            h = self.representation(obs_expand)

            next_h, r = self.dynamics(h, actions)
            _, next_v = self.prediction(next_h)

            r = -torch.log(1 - F.sigmoid(r) + 5e-3)

            next_v = support_to_scalar(next_v, self.support_size, self.support_step)

            assert len(next_v.shape) == 2, 'Next v error'.format(next_v.shape)
            assert len(r.shape) == 2, 'R shape error:{}'.format(r.shape)
            values = r + self.gamma * next_v

            return values.reshape(-1)

        else:
            assert False, 'Q Evaluation Assertion Error. Obs shape:{}, Action shape: {}'.format(obs.shape,
                                                                                                actions.shape)