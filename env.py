import gymnasium as gym
import numpy as np
from env import Environment


class Environment(gym.Env):
    def __init__(self, data):
        super(Environment, self).__init__()

        self.data = data
        self.int_columns = data.select_dtypes(include=[np.integer]).columns
        self.data = self.data[self.int_columns]
        self.n_features = len(self.int_columns)

        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.n_features),
            high=np.ones(self.n_features),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(2)
        self.current_state = 0

    def get_state(self):
        state = self.data.iloc[self.current_state].values
        return state

    def reset(self):
        self.current_state = 0
        return self.get_state()

    def step(self, action):
        if action == 1:
            reward = 1
        else:
            reward = -1
        self.current_state += 1
        done = self.current_state >= len(self.data) - 1
        next_state = self.get_state()   
        
        info = {
            "current_state": self.current_state,
            "reward": reward,
            "done": done
        }
        
        return next_state, reward, done, info

    def render(self, mode="human"):
        print(f"Current state: {self.current_state}")