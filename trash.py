from random import seed
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
import numpy as np
from trackmania import Trackmania
from tminterface.interface import TMInterface
import time
import signal
import csv
import joblib

# low_coords = np.array([51, 30, 48]).astype(np.int16)
# high_coords = np.array([55, 35, 73]).astype(np.int16)

# observation_space = spaces.Box(
#     low_coords, high_coords + 1, shape=(3,), dtype=np.int16, seed=123)
# action_space = spaces.Discrete(4)

# print(observation_space.high)
# print(observation_space.low)


# Initialize Q table
Q = joblib.load("./output_dicts/q_table_5000.pkl")
print(Q.shape)
print(Q[0, 0, 0])
print(np.argmax(Q[0, 0, 0]))
