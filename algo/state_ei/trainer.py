import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from algo.state_ei.model import MLPModel
from algo.state_ei.player import Player
import torch_utils
from torch_utils import *
import os
import ray
import time
import copy
from tensorboardX import SummaryWriter
from pickle_utils import *
import torch.autograd as autograd
import numpy as np

class Trainer:
    def __init__(self, checkpoint, config):
        self.model = MLPModel(config).to(torch.device('cuda'))
        self.model.set_weights(copy.deepcopy(checkpoint["weights"]))
        self.model.train()

        self.target_model = MLPModel(config).to(torch.device('cuda'))
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.eval()

        self.target_weight = copy.deepcopy(self.model.get_weights())
        self.selfplay_weight = copy.deepcopy(self.model.get_weights())

        self.player = Player(config, config.seed, config.num_workers)

        # self.target_model = MLPModel(config).to(torch.device('cuda'))

        if config.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=config.lr_init,
                                       weight_decay=config.weight_decay,
                                       momentum=config.momentum)
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=config.lr_init,
                                        weight_decay=config.weight_decay)

        if config.optimizer == 'Adam':
            self.scheduler = None
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     self.optimizer,
            #     T_max=120000
            # )
        else:
            self.scheduler = None

        self.config = config
        self.training_step = 0

        self.train_debug_batch = []

    def loss_consistency_atomic_fn(self, h, h_t):
        h = self.model.project(h, with_grad=True)
        h_t = self.model.project(h_t, with_grad=False)
        h = F.normalize(h, p=2., dim=-1, eps=1e-5)
        h_t = F.normalize(h_t, p=2., dim=-1, eps=1e-5)
        loss = -(h * h_t).sum(dim=1, keepdim=True)
        return loss

    def loss_reward_fn(self, r_pi_logits, r_exp_logits):
        loss = torch.cat((- F.logsigmoid(-r_pi_logits),
                          - F.logsigmoid(r_exp_logits)), dim=0)

        return loss

    def loss_value_fn(self, v, target_v):
        target_v = scalar_to_support(target_v, self.config.support_size, self.config.support_step).squeeze()
        loss = (-target_v * torch.nn.LogSoftmax(dim=1)(v)).sum(1)
        return loss

    def loss_pi_kl_fn(self, policy, target_action, target_policy):
        action_dim = policy.size(1) // 2
        n_branches = target_policy.size(1)

        target_action = target_action.clip(-0.999, 0.999)
        distr = SquashedNormal(policy[:, :action_dim], policy[:, action_dim:].exp())

        log_probs = []
        for i in range(n_branches):
            log_prob = distr.log_prob(target_action[:, i]).sum(-1, keepdim=True)
            log_probs.append(log_prob)

        policy_log_prob = torch.cat(log_probs, dim=1)
        loss = (-target_policy * policy_log_prob).sum(1)

        ent_action = distr.rsample()
        ent_action = ent_action.clip(-0.999, 0.999)
        ent_log_prob = distr.log_prob(ent_action).sum(-1, keepdim=True)
        entropy = - ent_log_prob.mean()

        return loss, entropy

    def loss_bc(self, policy, action):
        action_dim = policy.size(1) // 2
        action = action.clip(-0.999, 0.999)
        distr = SquashedNormal(policy[:, :action_dim], policy[:, action_dim:].exp())

        log_prob = distr.log_prob(action).sum(-1)
        return -log_prob

    def continuous_update_weights(self, batch_buffer, replay_buffer, target_workers, shared_storage):
        writer = SummaryWriter(self.config.results_path)
        num_played_games = 0

        print("Begin Initial Rollouts.")
        self.player.set_weights(copy.deepcopy(self.model.get_weights()))
        game_histories = self.player.run(0)
        for game_history in game_histories:
            replay_buffer.save_game.remote(game_history, shared_storage)

        num_played_games += len(game_histories)
        print("End Initial Rollouts.")

        self.player.set_eval_weights(self.model.get_weights())
        _, eval_reward = self.player.run_eval()
        print("Eval Finished.")

        target_counter = 0
        selfplay_counter = 0

        print("Launch Training !")

        # Training loop
        while self.training_step < self.config.training_steps:

            batch = batch_buffer.pop()
            if batch is None or batch == -1:
                time.sleep(0.3)
                continue

            index_batch, batch_step, batch = batch
            self.update_lr()

            x = time.time()
            (
                priorities,
                total_loss,
                gail_r,
                bootstrap_v,
                value_loss,
                reward_loss,
                first_step_reward_loss_pi,
                first_step_reward_loss_exp,
                policy_loss,
                consistency_loss,
                entropy_loss,
                grad_loss,
                bc_loss
            ) = self.update_weights(batch)

            print("Training Step:{}; Batch Step:{}; P:{:.3f}, V:{:.3f}, R:{:.3f}, FRP:{:.3f}, FRE:{:.3f}, C:{:.3f}, BC:{:.3f}, E:{:.3f}, G:{:.3f} "
                  "| GMax:{:.3f}, GMean:{:.3f}, GMin:{:.3f} | BVMax:{:.3f}, BVMean:{:.3f}, BVMin:{:.3f} | Time:{:.3f}".format(
                self.training_step, batch_step, policy_loss, value_loss, reward_loss, first_step_reward_loss_pi,
                first_step_reward_loss_exp, consistency_loss, bc_loss, entropy_loss, grad_loss,
                float(gail_r.max()), float(gail_r.mean()), float(gail_r.min()),
                float(bootstrap_v.max()), float(bootstrap_v.mean()), float(bootstrap_v.min()),
                time.time() - x
            ))

            target_counter += 1
            selfplay_counter += 1

            if target_counter > self.config.target_update_interval:
                self.target_weight = copy.deepcopy(self.model.get_weights())
                target_counter = 0
                self.target_model.set_weights(self.model.get_weights())

                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "target_weights": copy.deepcopy(self.target_weight),
                        "target_step": self.training_step
                    }
                )


            if selfplay_counter > self.config.selfplay_update_interval:
                self.selfplay_weight = copy.deepcopy(self.model.get_weights())
                selfplay_counter = 0

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights())
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()

            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                    "entropy_loss": entropy_loss,
                    "consistency_loss": consistency_loss
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)

            if self.config.ratio_lower_bound:

                desired_played_games = (1 + self.training_step // (self.config.epoch_repeat *
                                                                   self.config.num_workers *
                                                                   self.config.max_moves)) * self.config.num_workers

                if (num_played_games < desired_played_games
                    and self.training_step < self.config.training_steps
                ):
                    shared_storage.set_info.remote("running_games", True)
                    print("Launching rollout at training step", self.training_step)
                    self.player.set_weights(copy.deepcopy(self.model.get_weights()))
                    game_histories = self.player.run(self.training_step)
                    for game_history in game_histories:
                        replay_buffer.save_game.remote(game_history, shared_storage)

                    num_played_games += len(game_histories)
                    shared_storage.set_info.remote("running_games", False)
                    print("End rollout. # of games played = ", num_played_games,  "Desired:", desired_played_games)


            # Logs.
            if self.training_step % 10 == 0:
                # Log the information.
                writer.add_scalar("Training/Total_loss",total_loss, self.training_step)
                writer.add_scalar("Training/Value_loss", value_loss, self.training_step)
                writer.add_scalar("Training/Reward_loss", reward_loss, self.training_step)
                writer.add_scalar("Training/Grad_loss", grad_loss, self.training_step)
                writer.add_scalar("Training/FirstStep_Reward_lossPi", first_step_reward_loss_pi, self.training_step)
                writer.add_scalar("Training/FirstStep_Reward_lossExp", first_step_reward_loss_exp, self.training_step)
                writer.add_scalar("Training/Gail_Reward_Mean", float(gail_r.mean()), self.training_step)
                writer.add_scalar("Training/Gail_Reward_Max", float(gail_r.max()), self.training_step)
                writer.add_scalar("Training/Gail_Reward_Min", float(gail_r.min()), self.training_step)
                writer.add_scalar("Training/Bootstrap_Value_Mean", float(bootstrap_v.mean()), self.training_step)
                writer.add_scalar("Training/Bootstrap_Value_Max", float(bootstrap_v.max()), self.training_step)
                writer.add_scalar("Training/Bootstrap_Value_Min", float(bootstrap_v.min()), self.training_step)
                writer.add_scalar("Training/Policy_loss", policy_loss, self.training_step)
                writer.add_scalar("Training/Entropy_loss", entropy_loss, self.training_step)
                writer.add_scalar("Training/Contrastive_loss", consistency_loss, self.training_step)
                writer.add_histogram("Gail_Reward", gail_r, self.training_step)
                writer.add_histogram("Boostrap_V", bootstrap_v, self.training_step)

                num_reanalyze_games = ray.get(shared_storage.get_info.remote("num_reanalysed_games"))
                writer.add_scalar("Worker/Reanalyzed_games", num_reanalyze_games, self.training_step)
                writer.add_scalar("Worker/Batch_buffer_size", batch_buffer.get_len(), self.training_step)

            if self.training_step % 1000 == 0:
                # Log reward.
                mean_reward = ray.get(shared_storage.get_info.remote("mean_training_reward"))
                mean_true_reward = ray.get(shared_storage.get_info.remote("mean_training_true_reward"))
                if mean_reward is not None:
                    writer.add_scalar("Training/Mean Training Reward", mean_reward, self.training_step)

                if mean_true_reward is not None:
                    writer.add_scalar("Training/Mean Training True Reward", mean_true_reward, self.training_step)

                # Log game stats.
                num_played_steps = ray.get(shared_storage.get_info.remote("num_played_steps"))
                # num_played_games = ray.get(shared_storage.get_info.remote("num_played_games"))
                writer.add_scalar("Worker/num_played_steps", num_played_steps, self.training_step)
                writer.add_scalar("Worker/num_played_games", num_played_games, self.training_step)

            if self.training_step % 2000 == 0:
                shared_storage.set_info.remote("running_games", True)
                self.player.set_eval_weights(self.model.get_weights())
                _, eval_reward = self.player.run_eval()
                print("Eval Finished.")
                writer.add_scalar("Training/Eval", eval_reward, self.training_step // self.config.epoch_repeat)
                shared_storage.set_info.remote("running_games", False)

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update_weights(self, batch):
        """
        Perform one training step.
        """
        (
            observation_batch,
            next_observation_batch,
            action_batch,
            target_value,
            # target_reward,
            target_mu_batch,
            target_std_batch,
            weight_batch,
            gradient_scale_batch,
            mask_batch,
            raw_action_batch,
            raw_policy_batch
        ) = batch

        batchsize = observation_batch.shape[0]

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = np.array(target_value[:batchsize//2], dtype="float32")
        priorities = np.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device

        if self.config.PER:
            weight_batch = torch.from_numpy(weight_batch.copy()).float().to(device)


        observation_batch = torch.from_numpy(observation_batch).float().to(device)  # [B,O_SHAPE]
        next_observation_batch = torch.from_numpy(next_observation_batch).float().to(device)  # [B, UNROLL + 1, OSHAPE]
        mask_batch = torch.from_numpy(mask_batch).float().to(device)  # [B, UNROLL + 1, ASHAPE]
        action_batch = torch.from_numpy(action_batch).float().to(device)  # [B, UNROLL + 1, A_SHAPE]
        target_mu_batch = torch.from_numpy(target_mu_batch).float().to(device)  # [UNROLL_R + 1, B, A_SHAPE]
        target_std_batch = torch.from_numpy(target_std_batch).float().to(device)  # [UNROLL_R + 1, B, A_SHAPE]

        # The value here is the bootstrapped value, to make a bellman target, we also need to add the gail reward.
        target_value = torch.from_numpy(target_value).float().to(device)  # [B, UNROLL_R + 1]

        # target_reward = torch.from_numpy(target_reward).float().to(device)  # [B, UNROLL + 1]
        gradient_scale_batch = torch.from_numpy(gradient_scale_batch).float().to(device)
        raw_action_batch = torch.from_numpy(raw_action_batch).float().to(device)
        raw_policy_batch = torch.from_numpy(raw_policy_batch).float().to(device)

        batchsize = observation_batch.size(0)

        bootstrap_v = torch_utils.tensor_to_numpy(target_value).reshape(-1)
        bootstrap_v_mean, bootstrap_v_max = target_value.mean().item(), target_value.max().item()

        # assert observation_batch.max() < 2, "We assume that the observation is already processed here."

        # We need to calculate the reward first.
        gail_rewards = []  # [r0
        target_hiddens = []  # [h0, h1, h2, ...]
        last_observation = next_observation_batch[:, 0]

        for i in range(1, action_batch.shape[1]):
            _, _, _, gail_calc_hidden = self.target_model.initial_inference(last_observation)
            _, gail_reward, _, _ = self.target_model.recurrent_inference(
                gail_calc_hidden, action_batch[:, i]
            )
            gail_reward = -F.logsigmoid(-gail_reward).reshape(batchsize, 1).detach()
            gail_rewards.append(gail_reward)
            target_hiddens.append(gail_calc_hidden)
            last_observation = next_observation_batch[:, i]

        gail_rewards = torch.cat(gail_rewards, dim=1)

        """
            Calculating loss functions.
        """

        bc_loss = 0
        value_loss = 0
        reward_loss = 0
        policy_loss = 0
        consistency_loss = 0
        policy_entropy_loss = 0
        gradient_penalty_loss = 0

        value, reward, policy_info, hidden_state = self.model.initial_inference(
            observation_batch
        )

        policy_info_mcts, policy_info_bc = torch.chunk(policy_info, 2, dim=-1)

        for j in range(self.config.td_steps):
            target_value[:batchsize // 2, 0:1] += (self.config.discount ** j) * gail_rewards[:batchsize // 2, j:j+1]

        value_loss += self.loss_value_fn(value[:batchsize//2], target_value[:batchsize//2, 0:1])
        policy_loss_0, entropy = self.loss_pi_kl_fn(policy_info_mcts[:batchsize//2],
                                                    raw_action_batch[0][:batchsize//2],
                                                    raw_policy_batch[0][:batchsize//2])

        policy_loss += policy_loss_0
        policy_entropy_loss -= entropy

        bc_loss += self.loss_bc(policy_info_bc[batchsize // 2:], action_batch[batchsize // 2:, 1])

        pred_value_scalar = torch_utils.tensor_to_scalar(
            torch_utils.support_to_scalar(value, self.config.support_size).squeeze()
        )

        priorities[:, 0] = np.abs(pred_value_scalar - target_value_scalar[:, 0]) ** self.config.PER_alpha

        first_step_reward_loss_pi = 0
        first_step_reward_loss_exp = 0

        for i in range(1, self.config.num_unroll_steps_reanalyze + 1):
            """
                Calculate prediction loss iteratively.
            """

            # We add gradient penalty here.
            gp_hidden_pi = hidden_state[:batchsize//2, :].detach()
            gp_action_pi = action_batch[:batchsize//2, i]
            gp_hidden_exp = hidden_state[batchsize//2, :].detach()
            gp_action_exp = action_batch[batchsize//2, i]

            alpha = torch.rand(batchsize//2, 1).cuda()
            interpolate_hidden = gp_hidden_pi * alpha + gp_hidden_exp * (1 - alpha)
            interpolate_action = gp_action_pi * alpha + gp_action_exp * (1 - alpha)

            interpolate = torch.cat((interpolate_hidden, interpolate_action), dim=1)
            interpolate = autograd.Variable(interpolate, requires_grad=True)

            output_interpolate = self.model.reward(interpolate[:, :int(gp_hidden_pi.shape[1])],
                                                   interpolate[:, int(gp_hidden_pi.shape[1]):])

            gradients = autograd.grad(outputs=output_interpolate,
                                      inputs=interpolate,
                                      grad_outputs=torch.ones(output_interpolate.size()).cuda(),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            gradient_penalty_loss += gradient_penalty

            value, reward, policy_info, hidden_state = self.model.recurrent_inference(
                hidden_state, action_batch[:, i]
            )

            policy_info_mcts, policy_info_bc = torch.chunk(policy_info, 2, dim=-1)

            bc_loss += self.loss_bc(policy_info_bc[batchsize // 2:], action_batch[batchsize // 2:, i + 1])

            if self.config.ssl_target:
                target_hidden = self.target_model.encode(next_observation_batch[:, i])
            else:
                target_hidden = target_hiddens[i]

            consistency_loss += (self.loss_consistency_atomic_fn(hidden_state, target_hidden)
                                 * mask_batch[:, i:(i + 1)]).squeeze()

            reward_step_loss = self.loss_reward_fn(r_pi_logits=reward[:batchsize//2, :],
                                                   r_exp_logits=reward[batchsize//2:, :])

            if i == 1:
                first_step_reward_loss_pi = reward_step_loss[:batchsize//2].mean().item()
                first_step_reward_loss_exp = reward_step_loss[batchsize//2:].mean().item()

            reward_loss += reward_step_loss

            hidden_state.register_hook(lambda grad: grad * 0.5)

            if i <= self.config.num_unroll_steps_reanalyze:
                policy_loss_i, entropy = self.loss_pi_kl_fn(policy_info_mcts[:batchsize//2],
                                                            raw_action_batch[i][:batchsize//2],
                                                            raw_policy_batch[i][:batchsize//2])

                policy_loss += policy_loss_i
                policy_entropy_loss -= entropy

                # Now we need to calculate the target value
                for j in range(self.config.td_steps):
                    target_value[:batchsize//2, i:i+1] += (self.config.discount ** j) * gail_rewards[:batchsize//2,
                                                                                     i + j:i + j+1]

                value_loss += self.loss_value_fn(value[:batchsize//2], target_value[:batchsize//2, i:i + 1])

                pred_value_scalar = torch_utils.tensor_to_scalar(
                    torch_utils.support_to_scalar(value[:batchsize//2], self.config.support_size).squeeze()
                )
                priorities[:, i] = np.abs(pred_value_scalar - target_value_scalar[:, i]) ** self.config.PER_alpha

        target_loss = self.config.policy_loss_coeff * policy_loss + \
                      self.config.value_loss_coeff * value_loss + \
                      self.config.reward_loss_coeff * reward_loss

        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            target_loss *= weight_batch

        loss = target_loss.mean() + self.config.entropy_loss_coeff * policy_entropy_loss.mean() \
                + self.config.consistency_loss_coeff * consistency_loss.mean() \
                + self.config.grad_loss_coeff * gradient_penalty_loss.mean()

        if self.training_step % self.config.bc_frequency == 0:
            loss += self.config.bc_coeff * bc_loss.mean()


        loss.register_hook(lambda grad: grad * (1 / self.config.num_unroll_steps))
        parameters = self.model.parameters()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.training_step += 1

        if self.training_step % self.config.save_interval == 0:
            torch.save(
                self.model.state_dict(), os.path.join(
                    self.config.results_path,
                    'model_{}.pth'.format(self.training_step)
                )
            )

        # Save model to the disk.
        return (
            priorities,
            # For log purpose
            loss.item(),
            torch_utils.tensor_to_numpy(gail_rewards[:batchsize//2, 0]).reshape(-1),
            bootstrap_v.reshape(-1),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            first_step_reward_loss_pi,
            first_step_reward_loss_exp,
            policy_loss.mean().item(),
            consistency_loss.mean().item(),
            policy_entropy_loss.mean().item(),
            gradient_penalty_loss.mean().item(),
            bc_loss.mean().item()
        )
