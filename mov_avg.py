import numpy as np
import gym
import matplotlib.pyplot as plt
import joblib

rewards = joblib.load(
    "./output_dicts/rewards_NEW_last.pkl")
mov_avg = joblib.load(
    "./output_dicts/avg_rewards_NEW_last.pkl")
mov_avg_500, mov_avg_100 = [], []

EPISODES = 5_000
epsilon = 0.8
min_eps = 0.0
reduction = (epsilon - min_eps)/EPISODES
eps_list = []

WINDOW_SIZE_1 = 500
WINDOW_SIZE_2 = 100

for k in range(0, EPISODES):

    if k+1 >= WINDOW_SIZE_1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = rewards[k+1 - WINDOW_SIZE_1: k+1]

        # Calculate the average of current window
        window_average = round(sum(window) / WINDOW_SIZE_1, 0)

        # Store the average of current
        # window in moving average list
        mov_avg_500.append(window_average)
    else:
        mov_avg_500.insert(0, np.nan)

    if k+1 >= WINDOW_SIZE_2:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = rewards[k+1 - WINDOW_SIZE_2: k+1]

        # Calculate the average of current window
        window_average = round(sum(window) / WINDOW_SIZE_2, 0)

        # Store the average of current
        # window in moving average list
        mov_avg_100.append(window_average)
    else:
        mov_avg_100.insert(0, np.nan)

    epsilon -= reduction
    eps_list.append(epsilon)

# # Plot Rewards
# plt.figure(figsize=(25, 7))
# plt.plot((np.arange(len(rewards)) + 1), rewards)
# plt.plot((np.arange(len(mov_avg)) + 1), mov_avg)
# plt.plot((np.arange(len(eps_list)) + 1), eps_list)
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# plt.title('Reward vs Episodes')
# # plt.savefig('output_dicts/rewards.jpg')
# plt.show()
# # plt.close()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot((np.arange(len(rewards)) + 1), rewards, label='reward')
ax1.plot((np.arange(len(mov_avg)) + 1), mov_avg, label='avg_1000')
ax1.plot((np.arange(len(mov_avg_500)) + 1),
         mov_avg_500, label='avg_500')
ax1.plot((np.arange(len(mov_avg_100)) + 1),
         mov_avg_100, label='avg_100')
# ax2.plot((np.arange(len(eps_list)) + 1), eps_list, 'black', linestyle='--')
plt.xlabel('Episodes')
ax1.set_ylabel('Reward')
ax2.set_ylabel('Epsilon')
plt.title('Reward vs Episodes')

# plt.savefig('output_dicts/rewards.jpg')

plt.show()
