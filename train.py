import numpy as np
import gym
import matplotlib.pyplot as plt
import joblib

from env import TrackmaniaEnv


def save_outputs(episode, rewards, moving_avg, q_table):
    joblib.dump(q_table, f'./output_dicts/q_table_lr02eps60_{episode + 1}.pkl')
    joblib.dump(rewards, f'./output_dicts/rewards_lr02eps60_{episode +1 }.pkl')
    joblib.dump(
        moving_avg, f'./output_dicts/avg_rewards_lr02eps60_{episode + 1}.pkl')

    # # Plot Rewards
    # plt.figure(figsize=(25, 7))
    # plt.plot((np.arange(len(rewards)) + 1), rewards)
    # plt.plot((np.arange(len(moving_avg)) + 1), moving_avg)
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward')
    # plt.title('Reward vs Episodes')
    # plt.savefig(f'output_dicts/rewards_{episode}.jpg')
    # plt.close()


# Import and initialize Trackmania Env
env = TrackmaniaEnv()
env.reset()

# Define Q-learning function


def QLearning(env, learning, discount, epsilon, min_eps, episodes):

    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)
    # print(f"{num_states=}")
    num_states = np.round(num_states, 0).astype(int)

    # Initialize Q table
    # Q = np.zeros((num_states[0], num_states[1], num_states[2],
    #              num_states[3], env.action_space.n), dtype=np.int16)
    # print(f"{Q=}")
    # print(Q.shape)
    Q = np.zeros((num_states[0], num_states[1],
                 env.action_space.n), dtype=np.int16)

    # Initialize variables to track rewards
    reward_list = []
    moving_averages_list = []

    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/(episodes * 1)

    # Run Q learning algorithm
    for i in range(episodes):

        # Initialize parameters
        done = False
        total_reward, reward = 0, 0
        state = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low)
        state_adj = np.round(state_adj, 0).astype(int)

        while done != True:

            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(
                    Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Guarantee that the state is inside the obs space boundaries
            for k in range(len(state2)):
                if state2[k] < env.observation_space.low[k]:
                    state2[k] = env.observation_space.low[k]
                elif state2[k] >= env.observation_space.high[k]:
                    state2[k] = env.observation_space.high[k] - 1

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)
            state2_adj = np.round(state2_adj, 0).astype(int)

            # Allow for terminal states
            if done:
                Q[state_adj[0], state_adj[1], action] = reward

            # Adjust Q value for current state
            else:
                delta = learning*(reward + discount * np.max(
                    Q[state2_adj[0], state2_adj[1]]) - Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            # Update variables
            # total_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        if i < episodes * 1:
            if epsilon > min_eps:
                epsilon -= reduction

        # Track rewards
        reward_list.append(reward)

        # Loop through the array to consider
        # every window of size 100
        WINDOW_SIZE = 500

        if i+1 >= WINDOW_SIZE:
            # Store elements from i to i+window_size
            # in list to get the current window
            window = reward_list[i+1 - WINDOW_SIZE: i+1]

            # Calculate the average of current window
            window_average = round(sum(window) / WINDOW_SIZE, 0)

            # Store the average of current
            # window in moving average list
            moving_averages_list.append(window_average)
        else:
            moving_averages_list.insert(0, np.nan)

        if (i+1) % WINDOW_SIZE == 0:
            print('Episode {} Moving Average Reward: {}'.format(
                i+1, window_average))

        # SAVE_EVERY = 1_000
        # if (i+1) % SAVE_EVERY == 0:
        #     save_outputs(i, reward_list, moving_averages_list, Q)

    env.close()

    return Q, reward_list, moving_averages_list


# Run Q-learning algorithm
q_table, rewards, moving_avg = QLearning(env, 0.05, 0.9, 0.8, 0.0, 1_000)

joblib.dump(q_table, './output_dicts/q_table_lr005_last.pkl')
joblib.dump(rewards, './output_dicts/rewards_lr005_last.pkl')
joblib.dump(moving_avg, './output_dicts/avg_rewards_lr005_last.pkl')

# Plot Rewards
plt.plot((np.arange(len(rewards)) + 1), rewards)
plt.plot((np.arange(len(moving_avg)) + 1), moving_avg)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward vs Episodes')
plt.savefig('output_dicts/rewards.jpg')
plt.close()
