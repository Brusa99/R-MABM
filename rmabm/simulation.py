from pathlib import Path
from typing import Any, Callable

import numpy as np
from gymnasium.core import ObsType, ActType

from rmabm import Logger
from rmabm.agents import Dummy, QLearner
from rmabm.environments import Cats


class Simulation:
    """Wrapper for simulating the Cats environment and its interaction with agents.

    Args:
        environment: Cats environment to simulate.
        agents: List of agents that (externally) interact with the environment.
        n_episodes: Number of distinct episodes to simulate.
        current_episode: Current episode number. You may wish to not start from the first episode for logging purposes.

    Attributes:
        logger: Logger object for saving data (optional). If not set, data is not saved. Initialize with `init_logger`.

    Raises:
        ValueError: If the number of agents does not match the number of agents in the environment.

    """

    def __init__(
            self,
            environment: Cats,
            agents: list[QLearner | Dummy],
            n_episodes: int = 100,
            current_episode: int = 1,
    ):
        # check for valid input
        if environment.n_agents != len(agents):
            raise ValueError("Number of agents must match number of agents in the environment.")

        self.env = environment
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

    def run_episode(self, train: bool = False) -> None:
        """Run the simulation for one episode.

        A step is performed until the episode is terminated or truncated.
        After the episode, the `reset` method is called to prepare for the next episode.
        Current episode number is incremented by one.

        If a logger is initialized, the data is logged.

        Args:
            train: Whether to train the agents in this episode

        """
        # run the episode
        while not self._terminated and not self._truncated:
            self.step(train=train)

        # log data
        if self.logger:
            self.logger.log_obs(self._obs_history, self.current_episode)
            self.logger.log_act(self._act_history, self.current_episode)
            self.logger.log_array(np.array(self._reward_history), f"ep{self.current_episode}_rewards")
            self.logger.log_info(self._info_history, self.current_episode)

        self.reset()
        self.current_episode += 1

    def train_agents(self, n_episodes: int | None = None, update: Callable[..., None] | None = None) -> None:
        """Train the agents for a given number of episodes.

        Run a given number of episodes and train the agents at the end of each episode.
        If the number of episodes is not given, it will train until the current episode number reaches the number of
        episodes set at initialization.

        The method will automatically update the agents parameters at the end of each episode. It will use the provided
        update function where applicable (i.e.: it will be applied to no effect for Dummy agents).
        The function should have the following signature:
            `def update(agent, simulation) -> None`
        The `simulation` argument is the Simulation object that called the function (i.e.: self) and can be ignored. It
        can be used to use the simulation's information, like the current episode number or the total number of agents
        to update the agents' parameters.
        If the function is not provided, the agent's parameters will be updated as follows:
            - epsilon <- max(0.01, 0.9 ^ current_episode_number)
            - alpha is kept constant

        Args:
            n_episodes: Number of episodes to train the agents. Defaults to the number of episodes set at initialization
            update: Function to update the agents' parameters at the end of each episode. The function should accept the
                agent and the simulation as an argument and return None. If not provided, the agents' parameters are
                updated as described above.

        """
        n_episodes = n_episodes or (self.n_episodes - self.current_episode + 1)
        update = update or self._default_update

        for _ in range(n_episodes):
            self.run_episode(train=True)
            # update the agents
            for agent in self.qlearners:
                update(agent, self)

    @staticmethod
    def _default_update(agent: QLearner, simulation) -> None:
        agent.epsilon = max(0.01, 0.9 ** simulation.current_episode)

    @property
    def obs_history(self) -> list[list[ObsType]]:
        """Return the current episode's observations."""
        return self._obs_history

    @property
    def act_history(self) -> list[list[ActType]]:
        """Return the current episode's actions."""
        return self._act_history

    @property
    def reward_history(self) -> list[list[float]]:
        """Return the current episode's rewards."""
        return self._reward_history

    @property
    def info_history(self) -> list[dict[str, Any]]:
        """Return the current episode's infos."""
        return self._info_history
