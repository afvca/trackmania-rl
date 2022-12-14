
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


class TrackmaniaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, server_name="TMInterface0"):
        super().__init__()

        print(f'Connecting to {server_name}...')

        # Instanciate Trackmania client and set track name
        self.client = Trackmania(track="A0")

        # Start TMInterface server (this is equivalent to run_client method from TMInterface)
        self.iface = TMInterface(server_name)

        def handler(signum, frame):
            self.iface.close()
            quit()

        signal.signal(signal.SIGBREAK, handler)
        signal.signal(signal.SIGINT, handler)
        self.iface.register(self.client)

        while not self.iface.registered:
            time.sleep(0.1)

        # Set low and upper bounds of observation space
        # (X, Y, Z, Velocity) Coordinates
        # self.low_coords = np.array([500, 27, 480, 0]).astype(np.int16)  # *10
        # self.high_coords = np.array(
        #     [570, 35, 720, 250]).astype(np.int16)  # *10

        # self.observation_space = spaces.Box(
        #     self.low_coords, self.high_coords, shape=(4,), dtype=np.int16, seed=123)
        # self.action_space = spaces.Discrete(4)
        # self.steps = 0
        # # self.checkpoints_locations = self.checkpoints.copy()

        # self.low_coords = np.array([510, 30, 485]).astype(np.int16)
        # self.high_coords = np.array([545, 34, 714]).astype(np.int16)
        self.low_coords = np.array([51, 30, 48]).astype(np.int16)
        self.high_coords = np.array([56, 35, 73]).astype(np.int16)

        self.observation_space = spaces.Box(
            self.low_coords, self.high_coords, shape=(3,), dtype=np.int16, seed=123)
        self.action_space = spaces.Discrete(4)
        self.steps = 0
        # self.checkpoints_locations = self.checkpoints.copy()

    def step(self, action):

        observation = self.low_coords
        self.steps += 1
        reward = 0
        done = False
        info = {}

        # Block until client returns info
        while not self.client.info_ready:
            time.sleep(0)

        # Info ready from client
        if self.client.info_ready:

            # print(action)
            # training step returns an action that we need to send to the client
            self.client.action = action

            # state = [
            #     int(self.client.state_env[0]),  # *10
            #     int(self.client.state_env[1]),
            #     int(self.client.state_env[2]),  # *10
            #     int(self.client.state_env[3])
            # ]

            state = [
                int(round(self.client.state_env[0]/10, 0)),
                int(self.client.state_env[1]),
                int(round(self.client.state_env[2]/10, 0))
            ]

            # # Get the client state
            # state = [
            #     self.client.state_env[0] - self.checkpoints_locations[0][0],
            #     self.client.state_env[1] - self.checkpoints_locations[0][1],
            #     self.client.state_env[2] - self.checkpoints_locations[0][2],
            #     self.client.state_env[3]
            # ]
            # # breakpoint()

            # Get the reward from client
            reward = int(self.client.total_reward)

            # Unblock client to perform next action
            self.client.info_ready = False
            self.client.waiting_for_env = False

            # Check if client is in finish state (Arrived at finish line, fell off track, timeout track time...)
            done = self.client.finished

        observation = np.array(state, dtype=np.int16)
        # print(f"STEP {self.steps} - {state}")
        return observation, reward, done, info

    def reset(self):
        # print("resetting env...")
        self.steps = 0
        observation = [53, 34, 49]
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        print("closing env...")
        self.client.kill = False
        self.client.waiting_for_env = False

        self.iface.close()
        time.sleep(5)

        self.client = None
        self.iface = None


if __name__ == "__main__":
    env = TrackmaniaEnv()
    check_env(env)
    print("Environment checking complete!")
