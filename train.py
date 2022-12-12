import numpy as np
import gym
import matplotlib.pyplot as plt
import joblib


from env import TrackmaniaEnv

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
    Q = np.zeros((num_states[0], num_states[1], num_states[2], num_states[3], env.action_space.n), dtype=np.int16)
    # print(f"{Q=}")
    # print(Q.shape)

    # Initialize variables to track rewards
    reward_list = []
    avg_reward_list = []
    
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes
    
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
                action = np.argmax(Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Get next state and reward
            state2, reward, done, info = env.step(action)
            
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and state2[3] >= 7130:
                Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], action] = reward
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + discount * np.max(Q[state2_adj[0], state2_adj[1], state2_adj[2], state2_adj[3]]) - Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3],action])
                Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], action] += delta

            # Update variables
            # total_reward += reward
            state_adj = state2_adj
        
        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(reward)
        
        if (i+1) % 1 == 0:
            avg_reward = np.mean(reward_list)
            avg_reward_list.append(avg_reward)
            reward_list = []
            
        if (i+1) % 1 == 0:    
            print('Episode {} Average Reward: {}'.format(i+1, avg_reward))
            
    env.close()
    
    return Q, avg_reward_list

# Run Q-learning algorithm
q_table, rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 25)

joblib.dump(q_table, 'q_table_20.pkl')

# Plot Rewards
plt.plot((np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.jpg')     
plt.close()