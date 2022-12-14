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
import matplotlib.pyplot as plt

# low_coords = np.array([51, 30, 48]).astype(np.int16)
# high_coords = np.array([55, 35, 73]).astype(np.int16)

# observation_space = spaces.Box(
#     low_coords, high_coords + 1, shape=(3,), dtype=np.int16, seed=123)
# action_space = spaces.Discrete(4)

# print(observation_space.high)
# print(observation_space.low)


# # Initialize Q table
# Q = joblib.load("./output_dicts/q_table_5000.pkl")
# print(Q.shape)
# print(Q[3, :, 23])
# print(np.argmax(Q[0, 0, 0]))

epsilon = 0.8
min_eps = 0.05
reduction = (epsilon - min_eps)/(200-10)
eps_list = []
# Decay epsilon
for i in range(200):
    if i < 190:
        if epsilon > min_eps:
            epsilon -= reduction
    eps_list.append(epsilon)

# Plot Rewards
plt.plot((np.arange(len(eps_list)) + 1), eps_list)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward vs Episodes')
plt.show()
plt.close()
