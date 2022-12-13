import numpy as np
import gym
import matplotlib.pyplot as plt
import joblib

rewards = joblib.load("./output_dicts/rewards_5000.pkl")
mov_avg = joblib.load("./output_dicts/avg_rewards_5000.pkl")
moving_averages_list = []
WINDOW_SIZE = 100

epsilon = 0.8
min_eps = 0.05
reduction = (epsilon - min_eps)/5_000
eps_list = []

for k in range(0, 5000):
    if epsilon > min_eps:
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
ax1.plot((np.arange(len(rewards)) + 1), rewards)
ax1.plot((np.arange(len(mov_avg)) + 1), mov_avg)
ax2.plot((np.arange(len(eps_list)) + 1), eps_list, 'black', linestyle='--')
plt.xlabel('Episodes')
ax1.set_ylabel('Reward')
ax1.set_ylabel('Epsilon')
plt.title('Reward vs Episodes')
# plt.savefig('output_dicts/rewards.jpg')

plt.show()
