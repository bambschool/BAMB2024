import numpy as np
import gymnasium as gym

class CartPoleEnvironment:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.observation_shape = (20, 20, 20, 20)
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
        return discrete_state

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

# class GridWorld:
#     def __init__(self, size=5, starts=[(0,0)], goal=(4,4), obstacles=None, max_steps_per_trial=50):
#         self.size = size
#         self.starts = starts
#         self.goal = goal
#         self.obstacles = obstacles or []
#         self.max_steps_per_trial = max_steps_per_trial
#         self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
#         self.action_space = spaces.Discrete(4)
#         self.rewards = {'correct': 50, 'obstacle': -2, 'fail': -10, 'step_cost': -1}
#         self.position = self.starts[np.random.randint(len(self.starts))]
#         self.t = 0
#         self.trial = 0

#     def new_trial(self):
#         self.t = 0
#         self.trial += 1
#         self.position = self.starts[np.random.randint(len(self.starts))]

#     def step(self, action):
#         self.t += 1
#         new_trial = False
        
#         if action == 0:  # Move up
#             candidate = (max(self.position[0]-1, 0), self.position[1])
#         elif action == 1:  # Move right
#             candidate = (self.position[0], min(self.position[1]+1, self.size-1))
#         elif action == 2:  # Move down
#             candidate = (min(self.position[0]+1, self.size-1), self.position[1])
#         elif action == 3:  # Move left
#             candidate = (self.position[0], max(self.position[1]-1, 0))

#         if candidate == self.goal:
#             reward = self.rewards['correct']
#             new_trial = True
#             self.new_trial()
#         elif candidate in self.obstacles:
#             reward = self.rewards['obstacle']
#         else:
#             self.position = candidate
#             reward = self.rewards['step_cost']

#         if self.t > self.max_steps_per_trial:
#             new_trial = True
#             reward = self.rewards['fail']
#             self.new_trial()

#         return self.position, reward, {'new_trial': new_trial}

#     def render(self, ax=None):
#         grid = np.zeros((self.size, self.size))
#         grid[self.position] = 1
#         grid[self.goal] = 2
#         for obstacle in self.obstacles:
#             grid[obstacle] = -1
#         grid = np.pad(grid, pad_width=1, mode='constant', constant_values=-1)
        
#         if ax is None:
#             _, ax = plt.subplots()
#         ax.imshow(grid, vmin=-1, vmax=2, cmap='hot')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         return ax

# def plot_reward_per_episode(reward_per_episode, ax=None):
#     if ax is None:
#         _, ax = plt.subplots(figsize=(6,6))
#     ax.set_title("Reward per episode")
#     ax.set_xlabel("Episode")
#     ax.set_ylabel("Total Reward")
#     ax.plot(reward_per_episode)
#     return ax

# def plot_q_table(agent, action_space, max=50):
#     f, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,10))
#     ax = ax.flatten()
#     titles = ['Up', 'Right', 'Down', 'Left']
#     for i_a, a in enumerate(ax):
#         if i_a < len(action_space):
#             im = a.imshow(agent.get_q_table()[:, :, i_a], cmap='hot', vmin=-5, vmax=50)
#             plt.colorbar(im, ax=a)
#             a.set_title(titles[i_a])
#     plt.show()