import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
import copy

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        save_dir: str = None,
        load_models: bool = False,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.train_count = 0
        self.save_interval = 20
        self.save_dir = save_dir
        self.load_models = load_models

        if _init_setup_model:
            self._setup_model()

        # self.policy

        if self.load_models:
            self.policy.load_state_dict(th.load("saved_policies_20250310_04/policy_240.pth"))
            self.policy_2.load_state_dict(th.load("saved_policies_20250310_04/policy_2_240.pth"))

        self.optimizer = th.optim.AdamW(self.policy.parameters(), lr=learning_rate, fused=True, weight_decay=0.01)  # type: ignore[call-arg]
        self.optimizer_2 = th.optim.AdamW(self.policy_2.parameters(), lr=learning_rate, fused=True, weight_decay=0.01)  # type: ignore[call-arg]

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.policy_2.set_training_mode(True)
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)
        # self._update_learning_rate(self.policy_2.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        entropy_losses_2 = []
        pg_losses, value_losses = [], []
        pg_losses_2, value_losses_2 = [], []
        clip_fractions = []
        clip_fractions_2 = []

        min_target_kl = 0.005
        max_target_kl = 0.05

        # print(self.policy)

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            approx_kl_divs_2 = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # print("\n--- Training Mini-Batch ---") # Separator for mini-batch
                # print("Ratio (first 5 elements):\n", ratio.flatten()[:5]) # Print first few ratio values
                # print("Advantages (first 5 elements):\n", advantages[:5]) # Print first few advantage values


                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # policy_loss_2 = advantages * ratio
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # print("Policy Loss 1 (unclipped):\n", policy_loss_1.mean()) # Unclipped loss
                # print("Policy Loss 2 (clipped):\n", policy_loss_2.mean()) # Clipped loss
                # print("Policy Loss (min of clipped and unclipped):\n", policy_loss) # Final policy loss


                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    # print(log_ratio)
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 2.0 * self.target_kl:
                    continue_training = False
                    self.logger.record("player_0_train/early_stop_epoch", epoch)
                    # if self.verbose >= 1:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    self.target_kl *= 1.1
                    break
                elif self.target_kl is not None and approx_kl_div < 0.5 * self.target_kl:
                    self.target_kl *= 0.9

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                # self.policy.scheduler.step(loss.item())
            
            for rollout_data_2 in self.rollout_buffer_2.get(self.batch_size):
                actions_2 = rollout_data_2.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions_2 = rollout_data_2.actions.long().flatten()

                values_2, log_prob_2, entropy_2 = self.policy_2.evaluate_actions(rollout_data_2.observations, actions_2)
                values_2 = values_2.flatten()
                # Normalize advantage
                advantages_2 = rollout_data_2.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages_2) > 1:
                    advantages_2 = (advantages_2 - advantages_2.mean()) / (advantages_2.std() + 1e-8)
                
                # ratio between old and new policy, should be one at the first iteration
                ratio_2 = th.exp(log_prob_2 - rollout_data_2.old_log_prob)

                # clipped surrogate loss
                policy_loss_1_2 = advantages_2 * ratio_2
                policy_loss_2_2 = advantages_2 * th.clamp(ratio_2, 1 - clip_range, 1 + clip_range)
                # policy_loss_2_2 = advantages_2 * ratio_2
                policy_loss_2 = -th.min(policy_loss_1_2, policy_loss_2_2).mean()

                # Logging
                pg_losses_2.append(policy_loss_2.item())
                clip_fraction_2 = th.mean((th.abs(ratio_2 - 1) > clip_range).float()).item()
                clip_fractions_2.append(clip_fraction_2)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred_2 = values_2
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred_2 = rollout_data_2.old_values + th.clamp(
                        values_2 - rollout_data_2.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss_2 = F.mse_loss(rollout_data_2.returns, values_pred_2)
                value_losses_2.append(value_loss_2.item())

                # Entropy loss favor exploration
                if entropy_2 is None:
                    # Approximate entropy when no analytical form
                    entropy_loss_2 = -th.mean(-log_prob_2)
                else:
                    entropy_loss_2 = -th.mean(entropy_2)

                entropy_losses_2.append(entropy_loss_2.item())

                loss_2 = policy_loss_2 + self.ent_coef * entropy_loss_2 + self.vf_coef * value_loss_2

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417:
                # and discussion in PR #419:
                # and Schulman blog:
                with th.no_grad():
                    log_ratio_2 = log_prob_2 - rollout_data_2.old_log_prob
                    approx_kl_div_2 = th.mean((th.exp(log_ratio_2) - 1) - log_ratio_2).cpu().numpy()
                    approx_kl_divs_2.append(approx_kl_div_2)

                if self.target_kl is not None and approx_kl_div_2 > 2.0 * self.target_kl:
                    continue_training = False
                    self.logger.record("player_1_train/early_stop_epoch", epoch)
                    # if self.verbose >= 1:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div_2:.2f}")
                    self.target_kl *= 1.1
                    break
                elif self.target_kl is not None and approx_kl_div_2 < 0.5 * self.target_kl:
                    self.target_kl *= 0.9

                # Optimization step
                self.optimizer_2.zero_grad()
                loss_2.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy_2.parameters(), self.max_grad_norm)
                self.optimizer_2.step()
                # self.policy_2.scheduler.step(loss_2.item())

            # mean_kl = np.mean(approx_kl_divs + approx_kl_divs_2)

            # if mean_kl > 2.0 * self.target_kl:
            #     self.target_kl *= 0.9  # Reduce KL threshold if it's getting too high
            # elif mean_kl < 0.5 * self.target_kl:
            #     self.target_kl *= 1.1  # Increase KL threshold if updates are too small

            # self.target_kl = np.clip(self.target_kl, min_target_kl, max_target_kl)

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        explained_var_2 = explained_variance(self.rollout_buffer_2.values.flatten(), self.rollout_buffer_2.returns.flatten())

        # Logs
        self.logger.record("player_0_train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("player_0_train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("player_0_train/value_loss", np.mean(value_losses))
        self.logger.record("player_0_train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("player_0_train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("player_0_train/loss", loss.item())
        self.logger.record("player_0_train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("player_0_train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("player_0_train/n_updates", self._n_updates, exclude="tensorboard")

        # Logs
        self.logger.record("player_1_train/entropy_loss", np.mean(entropy_losses_2))
        self.logger.record("player_1_train/policy_gradient_loss", np.mean(pg_losses_2))
        self.logger.record("player_1_train/value_loss", np.mean(value_losses_2))
        self.logger.record("player_1_train/approx_kl", np.mean(approx_kl_divs_2))
        self.logger.record("player_1_train/clip_fraction", np.mean(clip_fractions_2))
        self.logger.record("player_1_train/loss", loss_2.item())
        self.logger.record("player_1_train/explained_variance", explained_var_2)
        if hasattr(self.policy_2, "log_std"):
            self.logger.record("player_1_train/std", th.exp(self.policy_2.log_std).mean().item())

        self.logger.record("player_1_train/n_updates", self._n_updates, exclude="tensorboard")

        self.logger.record("common_params/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("common_params/clip_range_vf", clip_range_vf)
        self.logger.record("common_params/adaptive_target_kl", self.target_kl)

        # print(self.policy.optimizer)
        # print(self.policy_2.optimizer)

        if self.train_count % self.save_interval == 0:
            th.save(self.policy.state_dict(), self.save_dir + f"policy_{self.train_count}.pth")
            th.save(self.policy_2.state_dict(), self.save_dir + f"policy_2_{self.train_count}.pth")
        #     if self.victor == "player_0":
        #         self.synchronize_models(self.policy, self.policy_2)
        #     else:
        #         self.synchronize_models(self.policy_2, self.policy)
        #     print("Models synchronized!")
        # self.victor = None

        self.train_count += 1
        self.policy, self.policy_2 = self.policy_2, self.policy
        self.rollout_buffer, self.rollout_buffer_2 = self.rollout_buffer_2, self.rollout_buffer
        self.optimizer, self.optimizer_2 = self.optimizer_2, self.optimizer
        # temp = copy.deepcopy(self.policy.state_dict())
        # self.policy.load_state_dict(self.policy_2.state_dict())
        # self.policy_2.load_state_dict(temp)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def synchronize_models(self, winner_model, loser_model):
        with th.no_grad():
            for p1, p2 in zip(winner_model.parameters(), loser_model.parameters()):
                p2.data.copy_(p1.data)
