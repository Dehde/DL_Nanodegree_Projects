import numpy as np
from task import Task
from scipy.interpolate import interp1d

class PolicySearch_Agent():
    def __init__(self, task, w2_size=100):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, w2_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range
        self.w2 = np.random.normal(
            size=(w2_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_w2 = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()
        self.clip_min = 300
        self.clip_max = 500

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        action = np.dot(action, self.w2)
        return self.map_intervalls(action)

    def perform(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.best_w)  # simple linear policy
        action = np.dot(action, self.best_w2)

        return self.map_intervalls(action)

    def map_intervalls(self, array_to_be_interpolated):
        map_func = interp1d([-1000,1000],[self.task.action_low,self.task.action_high])
        clipped_array = np.clip(array_to_be_interpolated, -1000, 1000)
        return map_func(clipped_array)


    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.best_w2 = self.w2
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.w2 = self.best_w2
            self.noise_scale = min(2.0 * self.noise_scale, 30.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        self.w2 = self.w2 + self.noise_scale * np.random.normal(size=self.w2.shape)  # equal noise in all directions
        