import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import time
from datetime import datetime

actions = [
    # Accelerate
    {
        "accelerate": True,
        "left": False,
        "right": False,
        "brake": False,
    },
    # Left
    {
        "accelerate": False,
        "left": True,
        "right": False,
        "brake": False,
    },
    # Right
    {
        "accelerate": False,
        "left": False,
        "right": True,
        "brake": False,
    },
    # Nothing
    {
        "accelerate": False,
        "left": False,
        "right": False,
        "brake": False,
    }
    # # Brake
    # {
    #     "accelerate": False,
    #     "left": False,
    #     "right": False,
    #     "brake": True,
    # },
    # # Accelerate Left
    # {
    #     "accelerate":True,
    #     "left":True,
    #     "right":False,
    #     "brake":False,
    # },
    # # Accelerate Right
    # {
    #     "accelerate":True,
    #     "left":False,
    #     "right":True,
    #     "brake":False,
    # },
    # # Brake Left
    # {
    #     "accelerate":False,
    #     "left":True,
    #     "right":False,
    #     "brake":True,
    # },
    # # Brake Right
    # {
    #     "accelerate":False,
    #     "left":False,
    #     "right":True,
    #     "brake":True,
    # }
]


def save_simulation_inputs(inputs, caption):

    if not os.path.exists("inputs"):
        os.makedirs("inputs")
    timestmp = time.time()
    date_time = datetime.fromtimestamp(timestmp)
    str_date_time = date_time.strftime("%d-%m-%Y_%H-%M-%S")
    with open(f"inputs/simulation_{caption}_{str_date_time}.txt", "w") as f:
        f.write(inputs)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True
