import sys
import time
import warnings
from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

import copy

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)

class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if _init_setup_model:
            self._setup_model()

        self.reward_rms = RunningMeanStd()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        self.rollout_buffer_2 = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

        # self.policy_2 = self.policy_class(  # type: ignore[assignment]
        #     self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        # )
        # self.policy_2 = self.policy_2.to(self.device)

        self.policy.mlp_extractor = th.compile(self.policy.mlp_extractor)
        # self.policy_2.mlp_extractor = th.compile(self.policy_2.mlp_extractor)

        self.policy_2 = copy.deepcopy(self.policy)


        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        """
        Recommend to use CPU only when using A2C/PPO with MlpPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        rollout_buffer_2: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        self.policy_2.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        rollout_buffer_2.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
            self.policy_2.reset_noise(env.num_envs)


        callback.on_rollout_start()

        # game_ended = [False for _ in range(env.num_envs)]
        game_ended = False

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
                self.policy_2.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs["player_0"], self.device)
                obs_tensor_2 = obs_as_tensor(self._last_obs["player_1"], self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                actions_2, values_2, log_probs_2 = self.policy_2(obs_tensor_2)
            actions = actions.cpu().numpy()
            actions_2 = actions_2.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            clipped_actions_2 = actions_2

            # if isinstance(self.action_space, spaces.Box):
            #     if self.policy.squash_output:
            #         # Unscale the actions to match env bounds
            #         # if they were previously squashed (scaled in [-1, 1])
            #         clipped_actions = self.policy.unscale_action(clipped_actions)
            #     else:
            #         # Otherwise, clip the actions to avoid out of bound error
            #         # as we are sampling from an unbounded Gaussian distribution
            #         clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            #     if self.policy_2.squash_output:
            #         # Unscale the actions to match env bounds
            #         # if they were previously squashed (scaled in [-1, 1])
            #         clipped_actions_2 = self.policy_2.unscale_action(clipped_actions_2)
            #     else:
            #         # Otherwise, clip the actions to avoid out of bound error
            #         # as we are sampling from an unbounded Gaussian distribution
            #         clipped_actions_2 = np.clip(actions_2, self.action_space.low, self.action_space.high)

            clipped_actions_copy = copy.deepcopy(clipped_actions)
            clipped_actions_2_copy = copy.deepcopy(clipped_actions_2)

            reshaped_actions = []
            action_rewards = []
            action_rewards_2 = []

            for i in range(clipped_actions_copy.shape[0]):
                reshaped_action = clipped_actions_copy[i].reshape(16, 3)
                reshaped_action[:, 1] = reshaped_action[:, 1] - 7
                reshaped_action[:, 2] = reshaped_action[:, 2] - 7

                attack_reward = self.attack_reward_func(reshaped_action, self._last_obs["player_0"]["unit_sap_range"][i][0], self._last_obs["player_0"]["enemy_visible_mask"][i])
                movement_reward = self.movement_reward_func(reshaped_action, self._last_obs["player_0"], i)
                action_reward = attack_reward + movement_reward
                action_rewards.append(action_reward)


                reshaped_action_2 = clipped_actions_2_copy[i].reshape(16, 3)
                reshaped_action_2[:, 1] = reshaped_action_2[:, 1] - 7
                reshaped_action_2[:, 2] = reshaped_action_2[:, 2] - 7

                attack_reward_2 = self.attack_reward_func(reshaped_action_2, self._last_obs["player_1"]["unit_sap_range"][i][0], self._last_obs["player_1"]["enemy_visible_mask"][i])
                movement_reward_2 = self.movement_reward_func(reshaped_action_2, self._last_obs["player_1"], i)
                action_reward_2 = attack_reward_2 + movement_reward_2
                action_rewards_2.append(action_reward_2)

                reshaped_action_all = {"player_0": reshaped_action, "player_1": reshaped_action_2}
                reshaped_actions.append(reshaped_action_all)

            reshaped_actions = np.array(reshaped_actions)

            new_obs, rewards, dones, infos = env.step(reshaped_actions)

            # print("original dones: ", dones)

            # dones_list = []
            # for done in dones:
            #     dones_list.append(done["player_0"] + done["player_1"])

            # dones = np.array(dones_list)

            # print("match_steps: ", self._last_obs["player_0"]["match_steps"][0])
            # print("next match_steps: ", new_obs["player_0"]["match_steps"][0])
            # print("steps: ", self._last_obs["player_0"]["steps"][0])
            # print("next steps: ", new_obs["player_0"]["steps"][0])
            # print("_last_episode_starts: ", self._last_episode_starts)
            # print("dones: ", dones)
            # print("last_episode_starts: ", self._last_episode_starts)
            # print("last_obs: ", self._last_obs["player_0"]["unit_positions"][0])

            # self.infos = infos

            # print(f"match_steps: {self._last_obs[0]["match_steps"]}, steps: {self._last_obs[0]["steps"]}")
            # print(f"next match_steps: {new_obs[0]["match_steps"]}, next steps: {new_obs[0]["steps"]}")
            # print(f"dones: {dones}")

            rewards_array = []
            for i in range(len(rewards)):
                total_rewards = rewards[i]["player_0"]
                total_rewards += self.env.map_explored_status_rewards[0]
                total_rewards += action_rewards[i]
                # total_rewards += self.env.relic_discovery_rewards[0] * (2500.0 * ((504 - self._last_obs["player_0"]["steps"][i][0]) ** np.e)/(504 ** np.e))
                # total_rewards += self.env.relic_discovery_rewards[0] * 2500.0 if self._last_obs["player_0"]["steps"][i][0] <= 100 else self.env.relic_discovery_rewards[0] * 500.0
                total_rewards += self.env.relic_discovery_rewards[0] * (2500.0 * ((504 - self._last_obs["player_0"]["steps"][i][0]) / 504))

                rewards_array.append(total_rewards)

            rewards_array = np.array(rewards_array)


            rewards_array_2 = []
            for i in range(len(rewards)):
                total_rewards_2 = rewards[i]["player_1"]
                total_rewards_2 += self.env.map_explored_status_rewards[1]
                total_rewards_2 += action_rewards_2[i]
                # total_rewards_2 += self.env.relic_discovery_rewards[1] * (2500.0 * ((504 - self._last_obs["player_1"]["steps"][i][0]) ** np.e)/(504 ** np.e))
                # total_rewards_2 += self.env.relic_discovery_rewards[1] * 2500.0 if self._last_obs["player_1"]["steps"][i][0] <= 100 else self.env.relic_discovery_rewards[1] * 500.0
                total_rewards_2 += self.env.relic_discovery_rewards[1] * (2500.0 * ((504 - self._last_obs["player_1"]["steps"][i][0]) / 504))

                rewards_array_2.append(total_rewards_2)

            rewards_array_2 = np.array(rewards_array_2)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            
            self._update_info_buffer(infos, dones)
            n_steps += 1

            # if isinstance(self.action_space, spaces.Discrete):
            #     # Reshape in case of discrete action
            #     actions = actions.reshape(-1, 1)
            #     actions_2 = actions_2.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    # print("timeout")
                    # terminal_obs = self.env.prepare_lux_obs(infos[idx]["terminal_observation"]["player_0"], 0,)
                    # terminal_obs_2 = self.env.prepare_lux_obs(infos[idx]["terminal_observation"]["player_1"], 1)
                    # print("total team points: ", infos[idx]["terminal_observation"]["player_0"]["total_team_points"])
                    # print("team wins: ", infos[idx]["terminal_observation"]["player_0"]["team_wins"])
                    # terminal_obs_all = obs_as_tensor([terminal_obs, terminal_obs_2], self.device)
                    print("total team points: ", self.env.total_team_points)
                    print("team wins: ", self.env.team_wins)
                    terminal_obs = infos[idx]["terminal_observation"]["player_0"]
                    terminal_obs_2 = infos[idx]["terminal_observation"]["player_1"]
                    self.logger.record("player_0_train/total points", self.env.total_team_points[0])
                    self.logger.record("player_1_train/total points", self.env.total_team_points[1])
                    terminal_obs_tensor = obs_as_tensor(terminal_obs, self.device)
                    terminal_obs_tensor_2 = obs_as_tensor(terminal_obs_2, self.device)
                    for key in terminal_obs_tensor.keys():
                        terminal_obs_tensor[key] = terminal_obs_tensor[key].unsqueeze(0)
                        terminal_obs_tensor_2[key] = terminal_obs_tensor_2[key].unsqueeze(0)
                    with th.no_grad():
                        # print(terminal_obs_tensor.keys())
                        # print(obs_tensor["enemy_energies"].shape)
                        # print(terminal_obs_tensor["enemy_energies"].shape)
                        terminal_value = self.policy.predict_values(terminal_obs_tensor)[0]  # type: ignore[arg-type]
                        terminal_value_2 = self.policy_2.predict_values(terminal_obs_tensor_2)[0]  # type: ignore[arg-type]
                    rewards_array[idx] += self.gamma * terminal_value
                    rewards_array_2[idx] += self.gamma * terminal_value_2

            # rewards_array = rewards_array / 30000.0
            # rewards_array_2 = rewards_array_2 / 30000.0

            # print(rewards_array.shape)
            # print(rewards_array_2.shape)
            # print(rewards_array)

            # rewards_array = np.clip(rewards_array, -1.0, 1.0)
            # rewards_array_2 = np.clip(rewards_array_2, -1.0, 1.0)

            # for i in range(env.num_envs):
            #     if self.env.game_over_flag_plus_one.sum() > 0:
            #         game_ended[i] = True

            # for i in range(env.num_envs):
            #     if game_ended[i]:
            #         rewards_array[i] = 0.0
            #         rewards_array_2[i] = 0.0

            #         values[i] = 0.0
            #         values_2[i] = 0.0
            if game_ended == False:
                if self.env.game_over_flag_plus_one:
                    game_ended = True
                    # print("game ended")

            if game_ended:
                rewards_array = np.zeros_like(rewards_array)
                rewards_array_2 = np.zeros_like(rewards_array_2)

                values = th.zeros_like(values, device=values.device)
                values_2 = th.zeros_like(values_2, device=values_2.device)

            # print("actions shape: ", actions.shape, actions)
            # print("rewards_array shape: ", rewards_array.shape, rewards_array)
            # print("values shape: ", values.shape, values)
            # print("log_probs shape: ", log_probs.shape, log_probs)

            rollout_buffer.add(
                self._last_obs["player_0"],  # type: ignore[arg-type]
                actions,
                rewards_array,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            rollout_buffer_2.add(
                self._last_obs["player_1"],  # type: ignore[arg-type]
                actions_2,
                rewards_array_2,
                self._last_episode_starts,  # type: ignore[arg-type]
                values_2,
                log_probs_2,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
        

        with th.no_grad():
            # Compute value for the last timestep
            new_obs_tensor = obs_as_tensor(new_obs["player_0"], self.device)
            new_obs_tensor_2 = obs_as_tensor(new_obs["player_1"], self.device)

            new_values = self.policy.predict_values(new_obs_tensor)  # type: ignore[arg-type]
            new_values_2 = self.policy_2.predict_values(new_obs_tensor_2)  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=new_values, dones=dones)
        rollout_buffer_2.compute_returns_and_advantage(last_values=new_values_2, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_2, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

            # self.policy, self.policy_2 = self.policy_2, self.policy
            # self.rollout_buffer, self.rollout_buffer_2 = self.rollout_buffer_2, self.rollout_buffer

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
    
    def attack_reward_func(self, actions, sap_range, enemy_unit_mask) -> float:

        attack_score = 0.0
        
        for i, action in enumerate(actions):
            # print(action)
            action_num, dx, dy = action[0], action[1], action[2]
            if action_num == 5:
                if enemy_unit_mask.sum() != 0:
                    pass
                    # sap_action_range = max(abs(dx), abs(dy))
                    # if sap_action_range > sap_range:
                    #     attack_score -= 0.5 / 16
                    # else:
                    #     attack_score += 1.0 / 16
                else:
                    attack_score -= 1.0 / 16
        
        # print(attack_score)
        
        return attack_score

    def next_position_calculator(self, action_num, unit_positions):
        # 0: stay, 1: up, 2: right, 3: down, 4: left

        if action_num == 1:
            next_position = (unit_positions[0], unit_positions[1] - 1)
        elif action_num == 2:
            next_position = (unit_positions[0] + 1, unit_positions[1])
        elif action_num == 3:
            next_position = (unit_positions[0], unit_positions[1] + 1)
        elif action_num == 4:
            next_position = (unit_positions[0] - 1, unit_positions[1])
        else:
            next_position = unit_positions
        
        return next_position

    def movement_reward_func(self, actions, obs, j) -> float:

        movement_score = 0.0

        for i, action in enumerate(actions):
            action_num, dx, dy = action[0], action[1], action[2]
            unit_positions = obs["unit_positions"][j][i]
            unit_energy = obs["unit_energies"][j][i]

            # # give penalty if try to move unit that doesn't exist
            # if (unit_positions == (-1, -1)).sum() == 2 and action_num != 0:
            #     movement_score -= 0.25 / 16
            
            # # give penalty if dx or dy is not 0 when not attacking
            # if action_num != 5:
            #     if dx != 0 or dy != 0:
            #         movement_score -= 0.25 / 16
            
            # if unit_positions[0] >= 0 and unit_positions[1] >= 0:
            #     # give penalty if try to move unit that has no energy
            #     if unit_energy <= 0 and action_num != 0:
            #         movement_score -= 0.25 / 16
            
            # give penalty if try to move unit out of map
            next_position = self.next_position_calculator(action_num, unit_positions)
            if next_position[0] < 0 or next_position[1] < 0 or next_position[0] > 23 or next_position[1] > 23:
                movement_score -= 0.5 / 16
            # else:
            #     movement_score += 1.0 / 16
        

        return movement_score
