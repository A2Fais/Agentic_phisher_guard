import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler


class Environment(gym.Env):
    def __init__(self, data):
        super(Environment, self).__init__()

        self.data_frame = data if isinstance(
            data, pd.DataFrame) else pd.DataFrame(data)

        exclude_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']
        self.feature_cols = [
            col for col in self.data_frame.columns if col not in exclude_cols]

        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(
            self.data_frame[self.feature_cols].values.astype(np.float32))

        self.y = self.data_frame["label"].values.astype(
            np.int32)  # target value (0 or 1)

        self.action_space = spaces.Discrete(2)

        n_features = len(self.feature_cols)

        self.observation_space = spaces.Box(
            low=np.zeros(n_features, dtype=np.float32),
            high=np.ones(n_features, dtype=np.float32),
            dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = len(self.X)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        return self.X[self.current_step], {}

    def step(self, action):
        true_label = self.y[self.current_step]
        reward = 1 if action == true_label else -1
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False 

        obs = self.X[self.current_step] if not terminated else np.zeros(
            len(self.feature_cols), dtype=np.float32)

        info = {"true_label": true_label} 
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.current_step < self.max_steps:
            print(f"Step: {self.current_step}, Features: {
                  self.X[self.current_step]}")
            self.print_scaled_example(self.current_step)
        else:
            print("Episode finished")

    def seed(self, seed=None):
        np.random.seed(seed)
