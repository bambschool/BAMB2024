from typing import Protocol
import numpy as np
import gymnasium as gym


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
