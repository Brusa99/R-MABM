from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class Logger:

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

        Args:
            array: Numpy array to save.
            filename: name of the file to save the array to. The file extension is automatically added.

        """
        # add file extension
        if not filename.endswith(".npy"):
            filename += ".npy"

        # path to array file
        file_path = self.path / filename

        # if array exists, append to file
        if file_path.exists():
            existing_array = np.load(file_path)
            array = np.concatenate([existing_array, array], axis=0)
            np.save(file_path, array)
        else:
            # create file and save array
            array.reshape(1, *array.shape)
            np.save(file_path, array)

    def log_dict(self, dictionary_list: list[dict[str, Any]]) -> None:
        """Wrapper for logging a list of dictionaries.

        This function saves the value for each key in a separate file given a list of dictionaries.
        The values are saved as numpy arrays, which allows to save space and load the data efficiently.

        Args:
            dictionary_list: List of dictionaries to log. The dictionaries must have the same keys.

        See Also:
            log_array: method called by this function to save the values of each key.

        """
        keys = dictionary_list[0].keys()
        # save values of each key in a separate file
        for key in keys:
            values = [d[key] for d in dictionary_list]
            self.log_array(np.array(values), str(key))
