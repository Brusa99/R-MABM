from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.core import ObsType, ActType


class Logger:
    """Utility class for logging data to files.

    This class is used to save data to files. It is particularly useful for saving data from simulations or experiments.
    It can save numpy arrays and dictionaries to files.

    Data is saved in the `./log_directory/log_name${TIMESTAMP}` directory.
    If the directory tree does not exist, it is created.
    Optionally, a timestamp can be added to the log name. This is useful to avoid overwriting previous logs.

    Args:
        log_name: Name of the log session.
        log_directory: Directory to save the log files. If the directory does not exist, it is created.
        use_timestamp: If True, a timestamp is added to the log name.

    """

    def __init__(self, log_name: str, log_directory: Path | str, use_timestamp: bool = True):
        # log session name
        self._log_name = log_name
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._log_name = f"{self._log_name}_{timestamp}"

        # log directory
        if not isinstance(log_directory, Path):
            log_directory = Path(log_directory)
        log_directory = log_directory / self._log_name  # append name to directory
        log_directory.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist

        # path = directory + name
        self.path = log_directory

    def log_array(self, array: np.ndarray, filename: str) -> None:
        """Save an array to a file.

        This function saves a numpy array to a file. If the file already exists, the array is appended to the file.
        If the array is empty, nothing is logged (no empty files are created).

        Args:
            array: Numpy array to save.
            filename: name of the file to save the array to. The file extension is automatically added.

        """
        # ignore empty arrays
        if array.size == 0:
            return

        # add file extension
        if not filename.endswith(".npy"):
            filename += ".npy"

        # path to array file
        file_path = self.path / filename

        # if array exists, append to file NOTE: there is no use case for this in the current implementation
        if file_path.exists():
            existing_array = np.load(file_path)
            array = np.concatenate([existing_array, array], axis=0)
            np.save(file_path, array)
        else:
            np.save(file_path, array)

    def log_info(self, info_list: list[dict[str, Any]], episode: int | None = None) -> None:
        """Wrapper for logging a list of dictionaries.

        This function saves the value for each key in a separate file given a list of dictionaries.
        The values are saved as numpy arrays, which allows to save space and load the data efficiently.
        If the lists are empty, like in the 0 agents case, the function does nothing.

        Args:
            info_list: List of dictionaries to log. The dictionaries must have the same keys.
            episode: Optional episode number to append to the array name.

        See Also:
            log_array: method called by this function to save the values of each key.

        """
        # check if the list, inner list and dictionary are empty
        if not info_list:
            return
        if not info_list[0]:
            return
        if not info_list[0].keys():
            return

        keys = info_list[0].keys()
        # save values of each key in a separate file
        for key in keys:
            values = [d[key] for d in info_list]
            if episode is not None:
                filename = f"ep{episode}_{key}"
            else:
                filename = str(key)
            self.log_array(np.array(values), filename)

    def _log_gym(self, gym_list: list[list[ObsType | ActType]], filename: str) -> None:
        """Wrapper for logging a list of lists of gymansium's actions or observations.

        This function takes in input a lists of lists of gymnasium's actions or observations, the inner list
        representing each agents action or observation, it unpacks them and then saves the unpacked numeric values as
        numpy arrays.
        If the lists are empty, like in the 0 agents case, the function does nothing.
        The actions of the dummy agents (which are not used) are saved as zeros. This is done to keep the array shape
        consistent across actions and observations.

        Args:
            gym_list: List of actions or observations to log.
            filename: name of the file to save the array to. The file extension is automatically added.

        See Also:
            log_array: method called by this function to save the values of each key.

        """
        # check if the list or inner list are empty
        if not gym_list:
            return
        if not gym_list[0]:
            return

        # calculate the dimension of an action or observation
        if isinstance(gym_list[0][0], dict):  # obs case
            gym_dim = len(gym_list[0][0].keys())
            gym_type = "obs"
        elif isinstance(gym_list[0][0], tuple):  # act case
            gym_dim = len(gym_list[0][0])
            gym_type = "act"
        else:
            raise TypeError("The gym_list must contain dictionaries or tuples.")

        # create an empty array to store ALL values
        shape = (
            len(gym_list),  # number of episodes
            len(gym_list[0]),  # number of agents
            gym_dim,  # number of elements in the observation or action
        )
        gym_array = np.empty(shape)

        # fill the array
        for step, gym_step in enumerate(gym_list):
            for agent, gym_agent in enumerate(gym_step):
                # set the values of the dummy agents to 0
                if "dummy" in gym_agent:
                    gym_array[step, agent, :] = 0
                else:
                    gym_array[step, agent, :] = list(gym_agent.values()) if gym_type == "obs" else gym_agent

        self.log_array(gym_array, filename)

    def log_act(self, act_list: list[list[ActType]], episode: int | None = None) -> None:
        """Wrapper for logging a list of lists of gymansium's actions.

        This function takes in input a lists of lists of gymnasium's actions, the inner list representing each agent's
        action, it unpacks them and then saves the unpacked numeric values as numpy arrays.
        If the lists are empty, like in the 0 agents case, the function does nothing.
        The actions of the dummy agents (which are not used) are saved as zeros. This is done to keep the array shape
        consistent across actions and observations. If only dummy actions are present, the function does nothing.

        Args:
            act_list: List of agents' actions or observations to log.
            episode: Optional episode number to append to the array name.

        See Also:
            log_array: method called by this function to save the values of each key.

        """
        # check if all actions are dummies
        if all(["dummy" in act for act in act_list[0]]):
            return

        filename = "act.npy"
        if episode is not None:
            filename = f"ep{episode}_{filename}"
        self._log_gym(act_list, filename)

    def log_obs(self, obs_list: list[list[ObsType]], episode: int | None = None) -> None:
        """Wrapper for logging a list of lists of gymansium's observations.

        This function takes in input a lists of lists of gymnasium's observations, the inner list representing each
        agent's observation, it unpacks them and then saves the unpacked numeric values as numpy arrays.
        If the lists are empty, like in the 0 agents case, the function does nothing.

        Args:
            obs_list: List of agents' observations to log.
            episode: Optional episode number to append to the array name.

        See Also:
            log_array: method called by this function to save the values of each key.

        """
        filename = "obs.npy"
        if episode is not None:
            filename = f"ep{episode}_{filename}"
        self._log_gym(obs_list, filename)
