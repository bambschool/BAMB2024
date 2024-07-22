from typing import Protocol
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv


class Environment(Protocol):
    def step(self, action) -> tuple:
        ...

    def reset(self) -> tuple:
        ...


class CartPoleEnvironment:
    def __init__(self, render_mode="rgb_array"):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.observation_shape = (20, 20, 20, 20)
        self.state_space_size = np.prod(self.observation_shape)
        self.action_space = self.env.action_space
        self.bins = self._create_bins()

    def _create_bins(self):
        bins = []
        for i in range(4):
            # For cart position and pole angle, use the environment's limits
            # For cart velocity and pole angular velocity, use -4 to 4 as limits
            low = self.env.observation_space.low[i] if (i == 0) or (i == 2) else -4
            high = self.env.observation_space.high[i] if (i == 0) or (i == 2) else 4

            # Create bins for each dimension of the state space
            item = np.linspace(low, high, num=self.observation_shape[i], endpoint=False)
            item = np.delete(item, 0)
            bins.append(item)
            # print(f"Bins for dimension {i}:\n{np.around(bins[i], 2)}\n")
        return bins

    def _get_discrete_state(self, state):
        discrete_state = tuple(np.digitize(state[i], self.bins[i]) for i in range(4))
        # print(f"Continuous state: {state}, Discrete state: {discrete_state}")
        return np.ravel_multi_index(discrete_state, self.observation_shape)

    def reset(self):
        state, _ = self.env.reset()
        return self._get_discrete_state(state), {}

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        discrete_next_state = self._get_discrete_state(next_state)
        return discrete_next_state, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class CustomGridEnv(MiniGridEnv):
    def __init__(
        self,
        agent_start_pos=(1, 1),
        goal_pos=(98, 98),
        size=100,
        max_steps=1000,
        render_mode="rgb_array"
    ):
        self.agent_start_pos = agent_start_pos
        self.goal_pos = goal_pos
        self.size = size

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            render_mode=render_mode,
        )

        # Override action space to include 9 actions
        self.action_space = spaces.Discrete(9)

        # Define the state space size
        self.state_space_size = size * size

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = 0  # Facing right

        # Place the goal
        self.put_obj(Goal(), *self.goal_pos)

        self.mission = self._gen_mission()

    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def step(self, action):
        self.step_count += 1

        direction = [
            (0, 0), (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]

        new_pos = (
            self.agent_pos[0] + direction[action][0],
            self.agent_pos[1] + direction[action][1]
        )

        if 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height:
            self.agent_pos = new_pos

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1 if done else 0

        state = self._get_state()

        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.agent_pos = self.agent_start_pos
        state = self._get_state()
        return state, {}

# Register the environment
gym.envs.registration.register(
    id='CustomGrid-v0',
    entry_point=CustomGridEnv,
)
