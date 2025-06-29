# TODO (stao): Add lux ai s3 env to gymnax api wrapper, which is the old gym api
import json
import os
from typing import Any, SupportsFloat
import flax
import flax.serialization
import gymnasium as gym
from gymnasium import spaces
# import gymnax
# import gymnax.environments.spaces
import jax
import numpy as np
import dataclasses
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.state import serialize_env_actions, serialize_env_states
from luxai_s3.utils import to_numpy
import random


class ModifiedLuxAIS3GymEnv(gym.Env):
    def __init__(self, numpy_output: bool = False):
        self.numpy_output = numpy_output
        self.rng_key = jax.random.key(random.randint(0, int(1e18)))
        self.jax_env = LuxAIS3Env(auto_reset=False)
        self.env_params: EnvParams = EnvParams()

        low = np.zeros((self.env_params.max_units, 3))
        low[:, 1:] = -self.env_params.unit_sap_range
        high = np.ones((self.env_params.max_units, 3)) * 6
        high[:, 1:] = self.env_params.unit_sap_range
        # self.action_space = gym.spaces.Dict(
        #     dict(
        #         player_0=gym.spaces.Box(low=low, high=high, dtype=np.int32),
        #         player_1=gym.spaces.Box(low=low, high=high, dtype=np.int32),
        #     )
        # )

        # self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.int32)

        # self.action_space = gym.spaces.Dict(
        #     dict(
        #         player_0=gym.spaces.MultiDiscrete([6, 15, 15] * self.env_params.max_units),
        #         player_1=gym.spaces.MultiDiscrete([6, 15, 15] * self.env_params.max_units)
        #     )
        # )

        self.action_space = gym.spaces.MultiDiscrete([6, 15, 15] * self.env_params.max_units)

        # self.observation_space = {
        #     "player_0": spaces.Dict({
        #         "units": spaces.Box(low=-1, high=23, shape=(self.env_params.max_units, 2), dtype=np.int32),
        #     })
        # }


        # **Observation Space (Dict)**
        self.observation_space = spaces.Dict({
            # **Own Units (Only visible units)**
            "unit_positions": spaces.Box(low=-1, high=23, shape=(self.env_params.max_units, 2), dtype=np.int32),
            "unit_energies": spaces.Box(low=-800, high=400, shape=(self.env_params.max_units,), dtype=np.int32),
            "unit_active_mask": spaces.MultiBinary(self.env_params.max_units),  # 1 if active, 0 if not
            # "unit_active_mask": spaces.Box(low=0, high=1, shape=(self.env_params.max_units,), dtype=np.int32),  # 1 if active, 0 if not
            
            # **Enemy Units (Only visible enemies)**
            "enemy_positions": spaces.Box(low=-1, high=23, shape=(self.env_params.max_units, 2), dtype=np.int32),
            "enemy_energies": spaces.Box(low=-800, high=400, shape=(self.env_params.max_units,), dtype=np.int32),
            "enemy_visible_mask": spaces.MultiBinary(self.env_params.max_units),  # 1 if visible, 0 if not
            # "enemy_visible_mask": spaces.Box(low=0, high=1, shape=(self.env_params.max_units,), dtype=np.int32),  # 1 if visible, 0 if not

            # **Team Vision (Sensor-based Fog of War)**
            "sensor_mask": spaces.MultiBinary((24, 24)),  # 1 where visible, 0 where fog-of-war applies
            # "sensor_mask": spaces.Box(low=0, high=1, shape=(24, 24,), dtype=np.int32),  # 1 where visible, 0 where fog-of-war applies

            # **Map Features (Masked by Vision)**
            "map_features_energy": spaces.Box(low=-7, high=10, shape=(24, 24), dtype=np.int32),  # Energy values
            "map_features_tile_type": spaces.Box(low=-1, high=2, shape=(24, 24), dtype=np.int32),  # Terrain types

            # **Relic Nodes & Points**
            "relic_nodes": spaces.Box(low=-1, high=23, shape=(6, 2), dtype=np.int32),  # Positions of relics
            "relic_nodes_mask": spaces.MultiBinary(6),  # 1 if tile gives relic points
            # "relic_nodes_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int32),  # 1 if tile gives relic points

            # **Game State Variables**
            "team_points": spaces.Box(low=0, high=2500, shape=(2,), dtype=np.int32),  # Team scores
            "team_wins": spaces.Box(low=0, high=3, shape=(2,), dtype=np.int32),  # Wins in best-of-5
            "steps": spaces.Box(low=0, high=505, shape=(1,), dtype=np.int32),  # Current step in match
            "match_steps": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),  # Total match steps

            ### Random parameters
            ## Given
            # unit_move_cost=list(range(1, 6)), # list(range(x, y)) = [x, x+1, x+2, ... , y-1]
            "unit_move_cost": spaces.Box(low=1, high=5, shape=(1,), dtype=np.int32),
            # unit_sensor_range=list(range(2, 5)),
            "unit_sensor_range": spaces.Box(low=2, high=4, shape=(1,), dtype=np.int32),
            # unit_sap_cost=list(range(30, 51)),
            "unit_sap_cost": spaces.Box(low=30, high=50, shape=(1,), dtype=np.int32),
            # unit_sap_range=list(range(3, 8)),
            "unit_sap_range": spaces.Box(low=3, high=7, shape=(1,), dtype=np.int32),
            ## Not given
            # # nebula_tile_vision_reduction=list(range(0,4)),
            # "nebula_tile_vision_reduction": spaces.Box(low=0, high=3, dtype=np.int32),
            # # nebula_tile_energy_reduction=[0, 0, 10, 25],
            # "nebula_tile_energy_reduction": spaces.Box(low=0, high=25, dtype=np.int32),
            # # unit_sap_dropoff_factor=[0.25, 0.5, 1],
            # "unit_sap_dropoff_factor": spaces.Box(low=0.25, high=1, dtype=np.float32),
            # "team_id": spaces.Discrete(2),  # 0 for player_0, 1 for player_1
            "team_id": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),  # 0 for player_0, 1 for player_1
            "my_spawn_location": spaces.Box(low=0, high=23, shape=(2,), dtype=np.int32),  # (x, y) coordinates
            "enemy_spawn_location": spaces.Box(low=0, high=23, shape=(2,), dtype=np.int32),  # (x, y) coordinates
            "map_explored_status": spaces.MultiBinary((24, 24)),  # 1 = Explored, 0 = Unexplored
            # "map_explored_status": spaces.Box(low=0, high=1, shape=(24, 24,), dtype=np.int32),  # 1 = Explored, 0 = Unexplored
        })


    def render(self):
        self.jax_env.render(self.state, self.env_params)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self.rng_key = jax.random.key(seed)
        self.rng_key, reset_key = jax.random.split(self.rng_key)
        # generate random game parameters
        # TODO (stao): check why this keeps recompiling when marking structs as static args
        randomized_game_params = dict()
        for k, v in env_params_ranges.items():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            randomized_game_params[k] = jax.random.choice(
                subkey, jax.numpy.array(v)
            ).item()
        params = EnvParams(**randomized_game_params)
        if options is not None and "params" in options:
            params = options["params"]

        self.env_params = params
        obs, self.state = self.jax_env.reset(reset_key, params=params)
        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))

        # only keep the following game parameters available to the agent
        params_dict = dataclasses.asdict(params)
        params_dict_kept = dict()
        for k in [
            "max_units",
            "match_count_per_episode",
            "max_steps_in_match",
            "map_height",
            "map_width",
            "num_teams",
            "unit_move_cost",
            "unit_sap_cost",
            "unit_sap_range",
            "unit_sensor_range",
        ]:
            params_dict_kept[k] = params_dict[k]
        return obs, dict(
            params=params_dict_kept, full_params=params_dict, state=self.state
        )

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.rng_key, step_key = jax.random.split(self.rng_key)
        obs, self.state, reward, terminated, truncated, info = self.jax_env.step(
            step_key, self.state, action, self.env_params
        )
        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))
            reward = to_numpy(reward)
            terminated = to_numpy(terminated)
            truncated = to_numpy(truncated)
            # info = to_numpy(flax.serialization.to_state_dict(info))
        return obs, reward, terminated, truncated, info


# TODO: vectorized gym wrapper


class RecordEpisode(gym.Wrapper):
    def __init__(
        self,
        env: ModifiedLuxAIS3GymEnv,
        save_dir: str = None,
        save_on_close: bool = True,
        save_on_reset: bool = True,
    ):
        super().__init__(env)
        self.episode = dict(states=[], actions=[], metadata=dict())
        self.episode_id = 0
        self.save_dir = save_dir
        self.save_on_close = save_on_close
        self.save_on_reset = save_on_reset
        self.episode_steps = 0
        if save_dir is not None:
            from pathlib import Path

            Path(save_dir).mkdir(parents=True, exist_ok=True)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if self.save_on_reset and self.episode_steps > 0:
            self._save_episode_and_reset()
        obs, info = self.env.reset(seed=seed, options=options)

        self.episode["metadata"]["seed"] = seed
        self.episode["params"] = flax.serialization.to_state_dict(info["full_params"])
        self.episode["states"].append(info["state"])
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.episode["states"].append(info["final_state"])
        self.episode["actions"].append(action)
        return obs, reward, terminated, truncated, info

    def serialize_episode_data(self, episode=None):
        if episode is None:
            episode = self.episode
        ret = dict()
        ret["observations"] = serialize_env_states(episode["states"])
        if "actions" in episode:
            ret["actions"] = serialize_env_actions(episode["actions"])
        ret["metadata"] = episode["metadata"]
        ret["params"] = episode["params"]
        return ret

    def save_episode(self, save_path: str):
        episode = self.serialize_episode_data()
        with open(save_path, "w") as f:
            json.dump(episode, f)
        self.episode = dict(states=[], actions=[], metadata=dict())

    def _save_episode_and_reset(self):
        """saves to generated path based on self.save_dir and episoe id and updates relevant counters"""
        self.save_episode(
            os.path.join(self.save_dir, f"episode_{self.episode_id}.json")
        )
        self.episode_id += 1
        self.episode_steps = 0

    def close(self):
        if self.save_on_close and self.episode_steps > 0:
            self._save_episode_and_reset()
