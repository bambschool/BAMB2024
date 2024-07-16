from typing import Protocol
import numpy as np


class Agent(Protocol):
    def act(self, state) -> int:
        ...

    def update_q_table(self, state, action, reward, next_state, next_action=None) -> None:
        ...


class RandomAgent:
    def __init__(self, action_space: int, state_space: int) -> None:
        # set up q table with correct dimensions
        self.action_space = action_space
        self.state_space = state_space

        # set up a random number generator
        self.rng = np.random.default_rng()

    def act(self, state) -> int:
        return self.rng.integers(0, self.action_space)

    def update_epsilon(self) -> None:
        pass

    def update_q_table(self, state, action, reward, next_state, next_action=None) -> None:
        pass


class SARSAAgent:
    def __init__(self, action_space: int, state_space: int) -> None:
        # set up q table with correct dimensions
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.zeros(shape=(state_space, action_space))

        # initialize agent's parameters
        self.gamma = 0.99
        self.alpha = 0.1
        self.epsilon = 1
        self.epsilon_decay = self.epsilon / 4_000

        # set up a random number generator
        self.rng = np.random.default_rng()

    def act(self, state) -> int:
        if self.rng.random() > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return self.rng.integers(0, self.action_space)

    def update_epsilon(self) -> None:
        if self.epsilon - self.epsilon_decay > 0:
            self.epsilon -= self.epsilon_decay

    def update_q_table(self, state, action, reward, next_state, next_action=None) -> None:
        current_q = self.q_table[state][action]

        if next_action is None:
            next_action = self.act(next_state)
        
        next_q = self.q_table[next_state][next_action]
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state][action] = new_q


class QLearningAgent:
    def __init__(self, action_space: int, state_space: int) -> None:
        # set up q table with correct dimensions
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.zeros(shape=(state_space, action_space))

        # initialize agent's parameters
        self.gamma = 0.99
        self.alpha = 0.1
        self.epsilon = 1
        self.epsilon_decay = self.epsilon / 4_000

        # set up a random number generator
        self.rng = np.random.default_rng()

    def act(self, state) -> int:
        if self.rng.random() > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return self.rng.integers(0, self.action_space)

    def update_epsilon(self) -> None:
        if self.epsilon - self.epsilon_decay > 0:
            self.epsilon -= self.epsilon_decay

    def update_q_table(self, state, action, reward, next_state, next_action=None) -> None:
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q