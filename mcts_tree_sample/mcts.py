import torch
import numpy as np
import mcts_tree_sample.cytree as tree
import torch_utils
import torch.nn.functional as F
import ray
from torch.cuda.amp import autocast as autocast


def run_multi(observations, model, config):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>

        _, root_reward, policy_info, roots_hidden_state = \
           model.initial_inference(torch_utils.numpy_to_tensor(observations))

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha]  * n_total_actions).astype(
            np.float32).tolist() for _ in range(root_num)]

        roots.prepare(config.root_exploration_fraction, noises, root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions)

            leaves_reward = leaves_reward.reshape(-1).tolist()
            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]


def run_multi_support(observations, model, config):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>

        _, root_reward, policy_info, roots_hidden_state = \
           model.initial_inference(torch_utils.numpy_to_tensor(observations))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config, True))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(
            np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]
        # print("ENTER", observations.shape)
        q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                torch_utils.numpy_to_tensor(root_actions))

        # print('Q_value_shape', q_values.shape)

        # During preparing, set Q_init.
        roots.prepare(config.root_exploration_fraction,
                      noises,
                      q_values.tolist(),
                      root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)

            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions)

            if leaves_reward.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config, False))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]



def run_multi_gail_support(observations, model, config):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>

        _, root_reward, policy_info, roots_hidden_state = \
           model.initial_inference(torch_utils.numpy_to_tensor(observations))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config, True))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha]  * n_total_actions).astype(
            np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]
        # print("ENTER", observations.shape)
        q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                torch_utils.numpy_to_tensor(root_actions))

        # print('Q_value_shape', q_values.shape)

        # During preparing, set Q_init.
        roots.prepare(config.root_exploration_fraction,
                      noises,
                      q_values.tolist(),
                      root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions)

            if leaves_reward.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            else:
                leaves_reward = -torch.log(1 - F.sigmoid(leaves_reward) + 1e-6)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size, config.support_step)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config, False))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]


def run_multi_gail_support_discrete(observations, model, config, transform=None):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION], CUDA_LONG_TENSORS>

        # _, root_reward, policy_info, roots_hidden_state = \
        #    model.initial_inference(torch_utils.numpy_to_tensor(observations))

        if transform is None:
            _, root_reward, policy_info, roots_hidden_state = \
                model.initial_inference(torch.from_numpy(observations).to('cuda').float())
        else:
            _, root_reward, policy_info, roots_hidden_state = \
                model.initial_inference(transform.transform(torch.from_numpy(observations).to('cuda').float()))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config, True))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha]  * n_total_actions).astype(
            np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]
        # print("ENTER", observations.shape)
        q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                torch_utils.numpy_to_tensor(root_actions).long())

        # print('Q_value_shape', q_values.shape)

        # During preparing, set Q_init.
        roots.prepare(config.root_exploration_fraction,
                      noises,
                      q_values.tolist(),
                      root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').long()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions.long())

            if leaves_reward.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            else:
                leaves_reward = -torch.log(1 - F.sigmoid(leaves_reward) + 1e-6)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size, config.support_step)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config, False))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]



def run_multi_gail_support_discrete_transform(observations, model, config, transform=None):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION], CUDA_LONG_TENSORS>

        if transform is None:
            _, root_reward, policy_info, roots_hidden_state = \
                model.initial_inference(torch.from_numpy(observations).to('cuda').float())
        else:
            _, root_reward, policy_info, roots_hidden_state = \
                model.initial_inference(transform.transform(torch.from_numpy(observations).to('cuda').float()))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config, True))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha]  * n_total_actions).astype(
            np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]
        # print("ENTER", observations.shape)
        q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                torch_utils.numpy_to_tensor(root_actions).long())

        # print('Q_value_shape', q_values.shape)

        # During preparing, set Q_init.
        roots.prepare(config.root_exploration_fraction,
                      noises,
                      q_values.tolist(),
                      root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').long()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions.long())

            if leaves_reward.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            else:
                leaves_reward = -torch.log(1 - F.sigmoid(leaves_reward) + 1e-6)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size, config.support_step)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config, False))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]

def run_multi_airl_support(observations, model, config):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>

        _, root_reward, policy_info, roots_hidden_state = \
           model.initial_inference(torch_utils.numpy_to_tensor(observations))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config, True))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha]  * n_total_actions).astype(
            np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]
        # print("ENTER", observations.shape)
        q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                torch_utils.numpy_to_tensor(root_actions))

        # print('Q_value_shape', q_values.shape)

        # During preparing, set Q_init.
        roots.prepare(config.root_exploration_fraction,
                      noises,
                      q_values.tolist(),
                      root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions)

            if leaves_reward.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            else:
                leaves_reward = leaves_reward #-torch.log(1 - F.sigmoid(leaves_reward) + 1e-6)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size, config.support_step)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config, False))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]


def run_multi_gailbc_support(observations, model, config):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>

        _, root_reward, policy_info, roots_hidden_state = \
           model.initial_inference(torch_utils.numpy_to_tensor(observations))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config, True))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples + config.mcts_num_bc_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha]  * n_total_actions).astype(
            np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]
        # print("ENTER", observations.shape)
        q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                torch_utils.numpy_to_tensor(root_actions))

        # print('Q_value_shape', q_values.shape)

        # During preparing, set Q_init.
        roots.prepare(config.root_exploration_fraction,
                      noises,
                      q_values.tolist(),
                      root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions)

            if leaves_reward.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            else:
                leaves_reward = -torch.log(1 - F.sigmoid(leaves_reward) + 1e-6)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size, config.support_step)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config, False))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]



def run_multi_gail_support_noise(observations, model, config, is_bootstrap=True):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>

        _, root_reward, policy_info, roots_hidden_state = \
           model.initial_inference(torch_utils.numpy_to_tensor(observations))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config, True, is_bootstrap))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha]  * n_total_actions).astype(
            np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]
        # print("ENTER", observations.shape)
        q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                torch_utils.numpy_to_tensor(root_actions))

        # print('Q_value_shape', q_values.shape)

        # During preparing, set Q_init.
        roots.prepare(config.root_exploration_fraction,
                      noises,
                      q_values.tolist(),
                      root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions)

            if leaves_reward.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            else:
                leaves_reward = -torch.log(1 - F.sigmoid(leaves_reward) + 1e-6)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size, config.support_step)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config, False, is_bootstrap))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]


def run_multi_gail_support_fast(observations, model, config, transform=None):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>


        if transform is None:
            _, root_reward, policy_info, roots_hidden_state = \
               model.initial_inference(torch.from_numpy(observations).to('cuda').float())
        else:
            _, root_reward, policy_info, roots_hidden_state = \
                model.initial_inference(transform.transform(torch.from_numpy(observations).to('cuda').float()))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config, True))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha]  * n_total_actions).astype(
            np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]
        # print("ENTER", observations.shape)

        with autocast():
            q_values = model.eval_q(torch.from_numpy(observations).cuda().float(),
                                    torch.from_numpy(root_actions).cuda().float())

        # print('Q_value_shape', q_values.shape)

        # During preparing, set Q_init.
        roots.prepare(config.root_exploration_fraction,
                      noises,
                      q_values.tolist(),
                      root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).cuda().float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).cuda().float()
            # print('SA', selected_actions.shape)

            with autocast():
                leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                    model.recurrent_inference(hidden_states, selected_actions)

            if leaves_reward.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            else:
                leaves_reward = -torch.log(1 - F.sigmoid(leaves_reward) + 1e-6)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size, config.support_step)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config, False))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]




def run_multi_gail2_support(observations, model, config):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>

        _, root_reward, policy_info, roots_hidden_state = \
           model.initial_inference(torch_utils.numpy_to_tensor(observations))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config, True))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha]  * n_total_actions).astype(
            np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]
        # print("ENTER", observations.shape)
        q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                torch_utils.numpy_to_tensor(root_actions))

        # print('Q_value_shape', q_values.shape)

        # During preparing, set Q_init.
        roots.prepare(config.root_exploration_fraction,
                      noises,
                      q_values.tolist(),
                      root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions)

            if leaves_reward.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            else:
                leaves_reward = -torch.log(1 - F.sigmoid(leaves_reward) + 1e-6)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size, config.support_step)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config, False))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]


def run_multi_support_ddpg_style(observations, model, config):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>

        _, root_reward, policy_info, roots_hidden_state = \
           model.initial_inference(torch_utils.numpy_to_tensor(observations))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config, True, clip=0.3))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(
            np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]
        # print("ENTER", observations.shape)
        q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                torch_utils.numpy_to_tensor(root_actions))

        # print('Q_value_shape', q_values.shape)

        # During preparing, set Q_init.
        roots.prepare(config.root_exploration_fraction,
                      noises,
                      q_values.tolist(),
                      root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions)

            if leaves_reward.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(leaves_policy, config, False, clip=0.3))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]


def run_multi_remote(observations, model, config):
    root_num = observations.shape[0]

    pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

    hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
    actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>

    _, root_reward, policy_info, roots_hidden_state = \
       ray.get(model.initial_inference.remote(observations))

    # root_reward shape          [256]
    # roots_hidden_state_shape = [[256, h]]

    hidden_states_pool.append(roots_hidden_state)
    actions_pool.append(ray.get(model.sample_mixed_actions.remote(policy_info, config)))
    hidden_state_idx_1 = 0

    n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
    roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
    noises = [np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(
        np.float32).tolist() for _ in range(root_num)]

    roots.prepare(config.root_exploration_fraction, noises, root_reward.reshape(-1).tolist())

    min_max_stats_lst = tree.MinMaxStatsList(root_num)
    min_max_stats_lst.set_delta(0.01)

    for index_simulation in range(config.num_simulations):
        hidden_states = []
        selected_actions = []
        results = tree.ResultsWrapper(root_num)
        data_idxes_0, data_idxes_1, last_actions = \
            tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

        ptr = 0
        for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
            hidden_states.append(hidden_states_pool[idx_1][idx_0])
            selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
            ptr += 1

        # hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
        # selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
        # print('SA', selected_actions.shape)

        leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
            ray.get(model.recurrent_inference.remote(hidden_states, selected_actions))

        leaves_reward = leaves_reward.reshape(-1).tolist()
        leaves_value = leaves_value.reshape(-1).tolist()

        # Update the database
        hidden_states_pool.append(leaves_hidden_state)
        actions_pool.append(ray.get(model.sample_mixed_actions.remote(leaves_policy, config)))
        hidden_state_idx_1 += 1

        # Back-propagate the reward information.
        tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                  leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]


