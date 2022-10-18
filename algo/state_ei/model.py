import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_utils
from torch_utils import *

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


class ResidualForwardModel(nn.Module):
    def __init__(self, s_shape, a_shape, dyn_model):
        super(ResidualForwardModel, self).__init__()
        self.mlp = mlp(s_shape + a_shape, dyn_model, s_shape, activation=nn.ReLU, use_bn=True)

    def forward(self, s, a):
        delta_s = self.mlp(torch.cat((s, a), dim=-1))
        s = s + delta_s
        return s


class MLPModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # print(config)
        self.gamma = config.discount
        self.support_size = config.support_size
        self.support_step = config.support_step

        self.obs_shape = config.mlp_obs_shape
        self.action_shape = config.mlp_action_shape
        self.hidden_shape = config.mlp_hidden_shape
        self.proj_shape = config.mlp_proj_shape
        self.n_stacked_obs = config.stacked_observations

        self.rep_net_shape = config.mlp_rep_shape
        self.rew_net_shape = config.mlp_rew_shape
        self.val_net_shape = config.mlp_val_shape
        self.pi_net_shape = config.mlp_pi_shape
        self.dyn_shape = config.mlp_dyn_shape
        self.proj_net_shape = config.mlp_proj_net_shape
        self.proj_pred_shape = config.mlp_proj_pred_net_shape

        self.reward_support_size = config.reward_support_size
        self.reward_support_step = config.reward_support_step
        self.full_reward_support_size = 2 * config.reward_support_size + 1

        '''
            Models
        '''
        if config.act == 'relu':
            act = nn.ReLU
        elif config.act == 'lrelu':
            print('Using LReLU')
            act = nn.LeakyReLU
        else:
            act = nn.ReLU

        if config.rep_act == 'tanh':
            rep_act = nn.Tanh
        else:
            rep_act = nn.Identity

        self.rep_net = mlp(self.obs_shape * self.n_stacked_obs, self.rep_net_shape, self.hidden_shape,
                           output_activation=rep_act, activation=act, use_bn=False)

        self.dyn_net = mlp(self.hidden_shape + self.action_shape, self.dyn_shape, self.hidden_shape,
                           output_activation=rep_act, activation=act, use_bn=False)

        # mlp(self.hidden_shape + self.action_shape, [64,], self.hidden_shape, activation=nn.Tanh, use_bn=False)
        self.rew_net = mlp(self.hidden_shape + self.action_shape, self.rew_net_shape, 1,
                           activation=act, use_bn=False, init_zero=config.init_zero)

        self.val_net = mlp(self.hidden_shape, self.val_net_shape, config.support_size * 2 + 1, activation=act,
                           use_bn=False, init_zero=config.init_zero)

        self.pi_net = mlp(self.hidden_shape, self.pi_net_shape, self.action_shape * 2,
                          activation=act, use_bn=False, init_zero=config.init_zero)

        self.pibc_net = mlp(self.hidden_shape, self.pi_net_shape, self.action_shape * 2,
                          activation=act, use_bn=False, init_zero=config.init_zero)

        self.proj_net = mlp(self.hidden_shape, self.proj_net_shape, self.proj_shape, use_bn=False)
        self.proj_pred_net = mlp(self.proj_shape, self.proj_pred_shape, self.proj_shape, use_bn=False)
        self.time_step = 0


    def discrim(self, hidden, action):
        return self.dis_net(hidden, action)

    def value(self, hidden):
        return self.val_net(hidden)

    def value_obs(self, obs):
        hidden = self.rep_net(obs)
        return self.value(hidden)

    def policy_no_split(self, hidden):
        pi = self.pi_net(hidden)
        # mu, log_std = pi[:, :self.action_shape], pi[:, self.action_shape:]
        return pi

    def policybc_no_split(self, hidden):
        pi = self.pibc_net(hidden)
        # mu, log_std = pi[:, :self.action_shape], pi[:, self.action_shape:]
        return pi

    def policy(self, hidden):
        pi = self.pi_net(hidden)
        mu, log_std = pi[:, :self.action_shape], pi[:, self.action_shape:]
        return mu, log_std

    def policy_obs(self, obs):
        hidden = self.rep_net(obs)
        return self.policy(hidden)

    def gail_reward(self, obs, action):
        return -torch.log(1 - F.sigmoid(self.reward(obs, action)) + 1e-6)

    def reward(self, hidden, action):
        # Only return logits.
        return self.rew_net(torch.cat((hidden, action), dim=1))

    def reward_obs(self, obs, action):
        hidden = self.rep_net(obs)
        return self.reward(hidden, action)

    def sample_mixed_actions(self, policy, config, is_root, is_bootstrap=False):
        n_batchsize = policy.shape[0]
        n_policy_action = config.mcts_num_policy_samples
        n_random_action = config.mcts_num_random_samples

        if config.explore_type == 'add':
            policy = policy.reshape(-1, policy.shape[-1])

            pi_mu, pi_logstd, exp_mu, exp_logstd = torch.chunk(policy, chunks=4, dim=-1)
            # mu, log_std = policy[:, :action_dim], policy[:, action_dim:]

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

    def eval_q(self, obs, actions):
        if len(obs.shape) == 2:
            if len(actions.shape) == 2:
                actions = actions.reshape(1, *actions.shape)

            # Obs shape = [BATCHSIZE, O_DIM]
            # Obs shape = [BATCHSIZE, N, A_DIM]

            batch_shape = obs.size(0)
            num_actions = actions.size(1)
            obs_expand = obs.reshape(obs.size(0), 1, obs.size(1)).repeat(1, actions.size(1), 1)
            obs_expand = obs_expand.reshape(obs_expand.size(0) * obs_expand.size(1), -1)
            actions = actions.reshape(actions.size(0) * actions.size(1), -1)

            h = self.encode(obs_expand)
            r = self.reward(h, actions)
            r = -torch.log(1 - F.sigmoid(r) + 1e-6)
            next_h = self.dynamics(h, actions)
            next_v = self.value(next_h)
            next_v = support_to_scalar(next_v, self.support_size, self.support_step)
            r = r.reshape(batch_shape, num_actions, 1)
            next_v = next_v.reshape(batch_shape, num_actions, 1)

            assert len(next_v.shape) == 3, 'Next v error'.format(next_v.shape)
            assert len(r.shape) == 3, 'R shape error:{}'.format(r.shape)
            values = r + self.gamma * next_v
            return values.squeeze()

        elif len(obs.shape) == 1:

            obs_expand = obs.reshape(1, -1).repeat(actions.size(0), 1)
            h = self.encode(obs_expand)
            r = self.reward(h, actions)
            r = -torch.log(1 - F.sigmoid(r) + 1e-6)
            next_h = self.dynamics(h, actions)
            next_v = self.value(next_h)
            next_v = support_to_scalar(next_v, self.support_size, self.support_step)

            assert len(next_v.shape) == 2, 'Next v error'.format(next_v.shape)
            assert len(r.shape) == 2, 'R shape error:{}'.format(r.shape)
            values = r + self.gamma * next_v

            return values.reshape(-1)

        else:
            assert False, 'Q Evaluation Assertion Error. Obs shape:{}, Action shape: {}'.format(obs.shape,
                                                                                                actions.shape)

    def encode(self, obs):
        return self.rep_net(obs)

    def dynamics(self, hidden, action):
        return self.dyn_net(torch.cat((hidden, action), dim=1))

    def project(self, h, with_grad=True):
        h_proj = self.proj_net(h)

        if with_grad:
            return self.proj_pred_net(h_proj)

        else:
            return h_proj.detach()

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def initial_inference(self, observation):
        # print(observation.shape)
        h = self.encode(observation)
        pi = self.policy_no_split(h)
        pibc = self.policybc_no_split(h)

        value = self.value(h)
        reward = torch.zeros(observation.size(0), 1).to(observation.device)

        pi_all = torch.cat((pi, pibc), dim=-1)

        return value, reward, pi_all, h

    def recurrent_inference(self, h, action):
        """

        :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
        :param action: shape: [Batchsize, Action_Dim]
        :return:
        """
        r = self.reward(h, action)
        h = self.dynamics(h, action)
        pi = self.policy_no_split(h)
        pibc = self.policybc_no_split(h)

        pi_all = torch.cat((pi, pibc), dim=-1)
        value = self.value(h)
        return value, r, pi_all, h