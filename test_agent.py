import numpy as np
import gym
import matplotlib.pyplot as plt
import joblib


from env import TrackmaniaEnv

# Import and initialize Trackmania Env
env = TrackmaniaEnv()
env.reset()

# Define Q-learning function


# def QLearning(env, qtable):

#     # Determine size of discretized state space
#     num_states = (env.observation_space.high - env.observation_space.low)
#     # print(f"{num_states=}")
#     num_states = np.round(num_states, 0).astype(int)

#     # Initialize Q table
#     Q = joblib.load(qtable)
#     print(Q.shape)

#     # Initialize variables to track rewards
#     reward_list = []
#     avg_reward_list = []

#     # Initialize parameters
#     done = False
#     total_reward, reward = 0, 0
#     state = env.reset()

#     # Discretize state
#     state_adj = (state - env.observation_space.low)
#     state_adj = np.round(state_adj, 0).astype(int)
#     print("state", state_adj)

#     while done != True:
#         action = np.argmax(Q[state_adj[0], state_adj[1], state_adj[2]])
#         print("action", action)

#         # print("state", state_adj)
#         # print("q-table", Q[state_adj[0], state_adj[1],
#         #       state_adj[2], state_adj[3]])
#         # print("action", action)

#         # Get next state and reward
#         state2, reward, done, info = env.step(action)

#         # Discretize state2
#         state2_adj = (state2 - env.observation_space.low)
#         state2_adj = np.round(state2_adj, 0).astype(int)

#         for k in range(len(state2_adj)):
#             if state2_adj[k] < env.observation_space.low[k]:
#                 state2_adj[k] = 0
#             elif state2_adj[k] > env.observation_space.high[k]:
#                 state2_adj[k] = env.observation_space.high[k] - \
#                     env.observation_space.low[k]

#         # Allow for terminal states
#         if done and state2[2] >= 713:
#             Q[state_adj[0], state_adj[1], state_adj[2], action] = reward

#         # Update variables
#         # total_reward += reward
#         state_adj = state2_adj

#     # Track rewards
#     reward_list.append(reward)

#     avg_reward = np.mean(reward_list)
#     avg_reward_list.append(avg_reward)
#     reward_list = []

#     print('Test Run. Average Reward: {}'.format(avg_reward))

#     env.close()

#     return Q, avg_reward_list


# # Run Q-learning algorithm
# q_table, rewards = QLearning(env, './output_dicts/q_table_5000.pkl')


def QLearning(env, qtable):

    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)
    # print(f"{num_states=}")
    num_states = np.round(num_states, 0).astype(int)

    # Initialize Q table
    Q = joblib.load(qtable)

    # Initialize variables to track rewards
    reward_list = []
    moving_averages_list = []

    # Initialize parameters
    done = False
    total_reward, reward = 0, 0
    state = env.reset()
    print("first state:", state)

    # Discretize state
    state_adj = (state - env.observation_space.low)
    state_adj = np.round(state_adj, 0).astype(int)

    while done != True:

        # Determine next action - epsilon greedy strategy
        action = np.argmax(Q[state_adj[0], state_adj[1], state_adj[2]])

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
        if done and state2[2] >= 71:
            Q[state_adj[0], state_adj[1], state_adj[2], action] = reward

        # Update variables
        state_adj = state2_adj

    # Track rewards
    reward_list.append(reward)
    print('Test Run. Average Reward: {}'.format(reward))

    env.close()

    return Q
