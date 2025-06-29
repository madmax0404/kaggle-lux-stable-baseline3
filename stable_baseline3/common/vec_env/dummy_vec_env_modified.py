import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info



class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: list[Callable[[], gym.Env]]):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.ids = ["player_0", "player_1"]

        self.buf_obs = OrderedDict([(id, OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])) for id in self.ids])
        # print(self.buf_obs)
        # self.buf_obs_2 = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        # self.buf_rews = np.zeros((self.num_envs, 2,), dtype=np.float32)
        self.buf_rews: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            # something = self.envs[env_idx].step(  # type: ignore[assignment]
            #     self.actions[env_idx]
            # )
            # print(something)
            # print(len(something))
            # print(self.buf_rews)
            # print(self.buf_infos)
            # obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(  # type: ignore[assignment]
            #     self.actions[env_idx]
            # )
            obs, reward, terminated, truncated, info = self.envs[env_idx].step(  # type: ignore[assignment]
                self.actions[env_idx]
            )
            self.game_over_flag_plus_one = obs["player_0"]["game_over_flag_plus_one"]
            self.total_team_points = obs["player_0"]["total_team_points"]
            self.team_wins = obs["player_0"]["team_wins"]
            obs["player_0"] = self.prepare_lux_obs(obs["player_0"], 0)
            obs["player_1"] = self.prepare_lux_obs(obs["player_1"], 1)

            # terminated = terminated["player_0"]
            # truncated = truncated["player_0"]

            self.buf_rews[env_idx] = reward
            self.buf_infos[env_idx] = info

            # print(rewa)
            # print(infooo)

            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # print("resetting")
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                obs["player_0"] = self.prepare_lux_obs(obs["player_0"], 0)
                obs["player_1"] = self.prepare_lux_obs(obs["player_1"], 1)

            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
            self.env_cfg = self.reset_infos[env_idx]["params"]
            self.spawn_location = np.array([[-1, -1], [-1, -1]], dtype=np.int32)
            self.map_explored_status = np.zeros((2, 24, 24), dtype=bool)
            self.first_spawn = False
            self.previous_map_explored_status_score = np.zeros((2, 1), dtype=np.int32)
            self.previous_map_explored_status_score[0] = self.map_explored_status[0].sum()
            self.previous_map_explored_status_score[1] = self.map_explored_status[1].sum()
            self.map_explored_status_rewards = np.zeros(2, dtype=np.int32)
            self.relic_discovery_rewards = np.zeros(2, dtype=np.int32)
            self.previous_relic_discovery_score = np.zeros(2, dtype=np.int32)

            obs["player_0"] = self.prepare_lux_obs(obs["player_0"], 0)
            obs["player_1"] = self.prepare_lux_obs(obs["player_1"], 1)

            # print(obs)


            self._save_obs(env_idx, obs)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)

    # def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
    #     player_0_obs = obs["player_0"]
    #     player_1_obs = obs["player_1"]

    #     self.game_over_flag_plus_one = player_0_obs["game_over_flag_plus_one"]

    #     player_0_lux_obs = self.prepare_lux_obs(player_0_obs, 0)
    #     player_1_lux_obs = self.prepare_lux_obs(player_1_obs, 1)

    #     for key in self.keys:
    #         if key is None:
    #             self.buf_obs[key][env_idx] = player_0_lux_obs
    #             self.buf_obs_2[key][env_idx] = player_1_lux_obs
    #         else:
    #             self.buf_obs[key][env_idx] = player_0_lux_obs[key]  # type: ignore[call-overload]
    #             self.buf_obs_2[key][env_idx] = player_1_lux_obs[key]  # type: ignore[call-overload]

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        # self.game_over_flag_plus_one = player_0_obs["game_over_flag_plus_one"]

        for i, id in enumerate(self.ids):
            for key in self.keys:
                if key is None:
                    self.buf_obs[id][key][env_idx] = obs[id]
                else:
                    
                    self.buf_obs[id][key][env_idx] = obs[id][key]  # type: ignore[call-overload]

    # def _obs_from_buf(self) -> VecEnvObs:
    #     return dict_to_obs(self.observation_space, deepcopy(self.buf_obs)), dict_to_obs(self.observation_space, deepcopy(self.buf_obs_2))
    
    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, deepcopy(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [env_i.get_wrapper_attr(attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [env_i.get_wrapper_attr(method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> list[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
    
    def prepare_lux_obs(self, obs, my_team_id):
        enemy_team_id = 1 - my_team_id

        if self.first_spawn == False:
            my_unit_positions = np.array(obs["units"]["position"][my_team_id])
            my_unit_mask = np.array(obs["units_mask"][my_team_id])
            my_available_unit_ids = np.where(my_unit_mask)[0]
            if my_available_unit_ids.shape[0] != 0:
                first_unit_id = my_available_unit_ids[0]
                first_unit_pos = my_unit_positions[first_unit_id]
                self.spawn_location[my_team_id] = (first_unit_pos[0], first_unit_pos[1])
                # self.enemy_spawn_location = self.find_opposite_corner_coords(self.map_explored_status[my_team_id], first_unit_pos[0], first_unit_pos[1])
                if self.spawn_location.sum() >= 0:
                    self.first_spawn = True

        self.map_explored_status[my_team_id][obs["map_features"]["tile_type"] != -1] = True

        model_input = {
            "enemy_energies": obs["units"]["energy"][enemy_team_id],
            "enemy_positions": obs["units"]["position"][enemy_team_id],
            "enemy_spawn_location": self.spawn_location[enemy_team_id],
            "enemy_visible_mask": obs["units_mask"][enemy_team_id],
            "map_explored_status": self.map_explored_status[my_team_id],
            "map_features_energy": obs["map_features"]["energy"],
            "map_features_tile_type": obs["map_features"]["tile_type"],
            "match_steps": np.array([obs["match_steps"]]),
            "my_spawn_location": self.spawn_location[my_team_id],
            "relic_nodes": obs["relic_nodes"],
            "relic_nodes_mask": obs["relic_nodes_mask"],
            "sensor_mask": obs["sensor_mask"],
            "steps": np.array([obs["steps"]]),
            "team_id": np.array([my_team_id]),
            "team_points": obs["team_points"],
            "team_wins": obs["team_wins"],
            "unit_active_mask": obs["units_mask"][my_team_id],
            "unit_energies": obs["units"]["energy"][my_team_id],
            "unit_move_cost": np.array([self.env_cfg["unit_move_cost"]]),
            "unit_positions": obs["units"]["position"][my_team_id],
            "unit_sap_cost": np.array([self.env_cfg["unit_sap_cost"]]),
            "unit_sap_range": np.array([self.env_cfg["unit_sap_range"]]),
            "unit_sensor_range": np.array([self.env_cfg["unit_sensor_range"]]),
        }

        # print("enemy spawn location", self.spawn_location[enemy_team_id])

        # print("match_steps", obs["match_steps"])
        # print("steps", obs["steps"])
        # print("team_points", obs["team_points"])
        # print("team_wins", obs["team_wins"])
        # print("map explored status", self.map_explored_status[my_team_id])

        current_map_explored_status_score = self.map_explored_status[my_team_id].sum()
        self.map_explored_status_rewards[my_team_id] = current_map_explored_status_score - self.previous_map_explored_status_score[my_team_id]
        if obs["steps"] <= 100:  # First 50 steps matter more
            self.map_explored_status_rewards[my_team_id] *= 5
        self.previous_map_explored_status_score[my_team_id] = current_map_explored_status_score

        current_relic_discovery_score = obs["relic_nodes_mask"].sum()
        self.relic_discovery_rewards[my_team_id] = current_relic_discovery_score - self.previous_relic_discovery_score[my_team_id]
        self.previous_relic_discovery_score[my_team_id] = current_relic_discovery_score

        # model_input = {k: torch.tensor(np.expand_dims(v, axis=0), dtype=torch.int32, device="cuda") for k, v in model_input.items()}

        return model_input
    
    def find_opposite_corner_coords(self, array, row, col):
        """
        Given a 2D array and a coordinate (row, col), this function returns the opposite corner coordinates.

        :param array: 2D list or NumPy array
        :param row: Row index of the given point
        :param col: Column index of the given point
        :return: (row', col') - Opposite corner coordinates
        """
        num_rows = len(array)
        num_cols = len(array[0]) if num_rows > 0 else 0

        # Opposite coordinates
        opp_row = num_rows - 1 - row
        opp_col = num_cols - 1 - col

        return (opp_row, opp_col)
