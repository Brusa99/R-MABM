from pathlib import Path
from typing import Any

import numpy as np

from pycats import Cats, QLearner, Dummy, Logger


class Simulation:
    """Wrapper for simulating the Cats environment and its interaction with agents.

    Args:
        env: Cats environment to simulate.
        agents: List of agents that (externally) interact with the environment.
        n_episodes: Number of distinct episodes to simulate.
        current_episode: Current episode number (default: 1).

    Attributes:
        logger: Logger object for saving data (optional). If not set, data is not saved. Initialize with `init_logger`.

    Raises:
        ValueError: If the number of agents does not match the number of agents in the environment.

    """

    def __init__(
            self,
            env: Cats,
            agents: list[QLearner | Dummy],
            n_episodes: int = 100,
            current_episode: int = 1,
    ):
        # check for valid input
        if env.n_agents != len(agents):
            raise ValueError("Number of agents must match number of agents in the environment.")

        self.env = env
        self.agents = agents
        self.qlearners = [agent for agent in agents if isinstance(agent, QLearner)]
        self.n_episodes = n_episodes
        self.current_episode = current_episode

        # logger for saving data (optional)
        self.logger: Logger | None = None

        # data_storage
        self._obs_history = []
        self._act_history = []
        self._reward_history = []
        self._info_history = []

        # reset environment and get current state
        self._obs, self._info = self.env.reset()
        self._next_obs = None
        self._reward = None
        self._terminated = False
        self._truncated = False

    def init_logger(self, log_name: str, log_directory: str | Path, use_timestamp: bool = True) -> None:
        """Initialize the logger for saving data.

        Args:
            log_name: Name of the log.
            log_directory: Directory to save the log to.
            use_timestamp: Whether to append a timestamp to the log name.

        """
        self.logger = Logger(log_name, log_directory, use_timestamp)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        """Reset the environment and internal histories and get the initial observation.

        Args:
            seed: Seed for the random number generator.
            options: Additional options for resetting the environment.

        """
        self._obs, self._info = self.env.reset(seed=seed, options=options)  # seed is setted here
        self._next_obs = None
        self._reward = None
        self._terminated = False
        self._truncated = False

        # reset histories
        self._obs_history = []
        self._act_history = []
        self._reward_history = []
        self._info_history = []

    def step(self, train: bool = False) -> None:
        """Run the simulation for one step.

        To perform a step, the agents' chosen action is selected given the current observations. The actions are passed
        to the environment, which returns the next observation, reward, and termination status. If enabled, the agents
        are then trained based on the reward and the next observation.
        Lastly, the observations are updated for the next step.

        Args:
            train: Whether to train the agents.

        """
        # get the actions from the agents
        actions = [agent.get_action(obs) for agent, obs in zip(self.agents, self._obs)]

        # take the actions in the environment
        self._next_obs, self._reward, self._terminated, self._truncated, self._info = self.env.step(actions)

        self._obs_history.append(self._obs)
        self._act_history.append(actions)
        self._reward_history.append(self._reward)
        self._info_history.append(self._info)

        # update the agents
        if train:
            for idx, agent in enumerate(self.agents):
                agent.train(self._obs[idx], self._reward[idx], self._next_obs[idx], self._terminated)

        # replace observation with next observation
        self._obs = self._next_obs

    def run_episode(self, train: bool = False, plot: bool = True, save_plot: str | None = None) -> None:
        """Run the simulation for one episode.

        A step is performed until the episode is terminated or truncated.
        After the episode, the `reset` method is called to prepare for the next episode.
        If a logger is initialized, the data is logged.
        If plotting is enabled, the results are plotted and optionally saved to the specified path. If no path is given,
        the plot is not saved.

        Args:
            train: Whether to train the agents in this episode
            plot: Whether to plot the results of the episode
            save_plot: Path to save the plot to. If None, the plot is not saved.

        """
        # run the episode
        while not self._terminated and not self._truncated:
            self.step(train=train)

        # log data
        if self.logger:
            self.logger.log_dict(self._obs_history)
            self.logger.log_dict(self._act_history)
            self.logger.log_array(np.array(self._reward_history), "rewards")
            self.logger.log_dict(self._info_history)

        # plot results
        if plot:
            pass  # TODO: implement plotting


