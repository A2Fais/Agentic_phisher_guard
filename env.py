import gymnasium as gym
import numpy as np


class Environment(gym.Env):
    def __init__(self, data):
        super(Environment, self).__init__()
        filtered_integer_columns = data.select_dtypes(
            include=[np.integer]).columns
        
        print(len(filtered_integer_columns), filtered_integer_columns)
