from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

import sys
import numpy as np
import csv
import time
import random

from utils import actions, save_simulation_inputs

#########
# GAME COORDINATE SYSTEM
# POSITION IN  x: self.state.position[0], y: self.state.position[1], z: self.state.position[2]
#########

class Trackmania(Client):

    def __init__(self, track: str) -> None:
        self.state              = None
        self.initial_state      = None
        self.finished           = False
        self.close_client       = False

        self.total_reward       = 0
        self.max_score          = 0
        self.max_race_time      = 30000
        self.best_time          = float("inf")
        self.race_time          = 0
        
        self.out_of_bounds      = 0
        self.cp_count           = 0
        self.sim_count          = 0

        self.action             = 0
        self.prev_action        = 0

        self.waiting_for_env    = False
        self.info_ready         = False

        self.checkpoints_locations = [0,0,0]
        # Load Checkpoints
        with open(f'checkpoints/TASI_{track}.csv', 'r') as read_obj:
            csv_reader = csv.reader(read_obj, quoting=csv.QUOTE_NONNUMERIC)
            next(csv_reader, None)
            self.checkpoints = list(csv_reader)

        super().__init__()

    def on_registered(self, iface: TMInterface) -> None:
        iface.execute_command('set controller none') # turn off bruteforce
        # iface.execute_command('set speed 10') # set race speed
        iface.execute_command('set sim_speed 5') # set simulation speed
        iface.set_timeout(200000) # timeout after 200 secs
        print(f'Registered to {iface.server_name}')

    def on_simulation_begin(self, iface: TMInterface):
        print("Started new simulation")
        iface.set_simulation_time_limit(self.max_race_time)
        iface.remove_state_validation()
        iface.clear_event_buffer()                ## Clears all inputs in current replay
        self.finished = False
        self.checkpoints_locations = self.checkpoints.copy()
        # iface.set_simulation_time_limit(30000)  ## tempo limite para completar a pista


    def on_simulation_step(self, iface: TMInterface, _time: int):
        self.race_time = _time

        ## INITIAL CLEAN-UP
        if self.race_time == -10:
            self.initial_state = iface.get_simulation_state()
        # if _time <= 0:
        #     iface.set_input_state(
        #         sim_clear_buffer=True
        #     )

        if self.race_time >= self.max_race_time:
            print("timeout")
            self.finished = True
            self.update_reward("timeout")
            # iface.prevent_simulation_finish()
            # self.restart_simulation(iface)

        if self.initial_state == None and _time == 0:
            self.initial_state = iface.get_simulation_state()

        # Step at each 200ms
        if (_time % 200 == 0 and _time >= 0):

            self.waiting_for_env = True
            self.state = iface.get_simulation_state()

            # Check if car has fallen off the track (Y coordinate, index 1 from position object)
            if self.state.position[1] < 30:
                # print("total reward:", self.total_reward)
                print("out of bounds")
                self.finished = True
                self.out_of_bounds = 1
                self.update_reward("out_of_bounds")
                # self.restart_simulation(iface)

            posx = self.state.position

            # If we have more than one checkpoint
            # See if we need to advance the checkpoint
            if len(self.checkpoints_locations) > 1:
                # compare the first remaining checkpoint to current Z coordinate of the car
                if np.round(self.checkpoints_locations[0][2], 0).astype(int) < np.round(posx[2], 0).astype(int): #*10
                    # if z coordinate of the car already passed the checkpoint
                    # remove the checkpoint from the current remaining cp's
                    # update the reward function to reflect the cp passage
                    self.checkpoints_locations.pop(0)
                    self.update_reward("checkpoint", len(self.checkpoints) - len(self.checkpoints_locations))

            velocity_kmh = np.linalg.norm(self.state.velocity) * 3.6
            self.update_reward("velocity", velocity_kmh)

            posx += [velocity_kmh]
            self.state_env = posx

            #### Info ready for the Gym environment
            self.info_ready = True
            start = time.time()
            while self.waiting_for_env:
                if time.time() - start > 10:
                    break
                pass

            # action = actions[0] 
            # action = actions[random.randint(1,3)]
            action = actions[self.action]

            self.current_action = {
                'sim_clear_buffer': False,
                "accelerate":   action["accelerate"],
                "left":         action["left"],
                "right":        action["right"],
                "brake":        action["brake"],
            }

            iface.set_input_state(**self.current_action)

        if self.finished or self.race_time >= self.max_race_time:
            if not self.close_client:
                self.restart_simulation(iface)
            self.finished = False


    def update_reward(self, event, value=0):
        # update reward if checkpoint is passed
        if event == 'checkpoint':
            # print("reward cp")
            # reward += .1 * (self.max_race_time - self.race_time)
            self.total_reward += value * 100
        elif event == 'finish':
            # print("finish")
            self.total_reward += 10_000 
        # update reward if distance driven
        elif event == 'velocity':
            self.total_reward += value * 1
        # update reward if out_of_bounds or timeout
        elif event == 'out_of_bounds' or event == 'timeout':
            self.total_reward -= 1_000
        
        ######
        # TODO: add more reward calculations... (dif tempo, distancia prox checkpoint)
        ######

    def restart_simulation(self, iface: TMInterface):
        
        print(f'[{self.sim_count+1}] Finished simulation at {self.race_time}. Reward: {self.total_reward}')

        # get simulation inputs to str
        inputs = iface.get_event_buffer().to_commands_str()
        if ((self.sim_count+1) % 5 == 0) or (self.max_score < self.total_reward):
            save_simulation_inputs(inputs, str(self.sim_count+1) + "_" + str(self.race_time) + "_" + str(int(self.total_reward)) + "_" + str(self.out_of_bounds) + "_" + str(self.finished))

        # check if current high score is less than simulation total reward
        if self.max_score < self.total_reward:
            # update agent max score
            self.max_score = self.total_reward

        # cleanup vars
        self.cp_count       = 0
        self.current_cp     = 0
        self.total_reward   = 0
        self.best_time      = float("inf")
        self.race_time      = 0
        self.out_of_bounds  = 0
        self.finished       = False
        self.sim_count      += 1
        self.checkpoints_locations = self.checkpoints.copy()

        # clear event buffer
        iface.clear_event_buffer()

        # restart simulation to initial state
        iface.rewind_to_state(self.initial_state)

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):

        # print(f"current CP: {current}; target CP: {target}; {iface.get_checkpoint_state().cp_times}")
        self.cp_count = len([time for (time, _) in iface.get_checkpoint_state().cp_times if time != -1])

        ## update reward by 100 after passing a checkpoint
        self.update_reward('checkpoint', self.cp_count)

        # print(self.cp_count)
        if current == target:
            ## update reward by 100 after passing a checkpoint
            self.update_reward('finish', 1000)
            # self.last_time = iface.get_checkpoint_state().cp_times[-1][0]
            # self.race_time = iface.get_simulation_state().race_time
            # print(f'[{self.sim_count}] Finished the race at {self.race_time}. Reward: {self.total_reward}')
            self.finished = True

            if not self.close_client:
                iface.prevent_simulation_finish()

    def on_simulation_end(self, iface, result: int):
        print('Simulation finished')


def main(track="A0"):
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(Trackmania(track), server_name)


if __name__ == '__main__':
    main()