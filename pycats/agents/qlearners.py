import warnings
from typing import Iterable, Callable

import numpy as np

from pycats.environments import Cats


class QLearner:
    def __init__(
            self,
            agent_id: int,
            environment: Cats,
            n_bins: int | tuple[int, int, int, int] = (11, 11, 11, 11),
            alpha: float = 0.1,
            gamma: float = 0.99,
            epsilon_zero: float = 0.9,
    ):
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_zero

        # fill in n_bins if only one value is given
        if isinstance(n_bins, int):
            n_bins = (n_bins, n_bins, n_bins, n_bins)

        # init Q table
        self.Q = np.zeros(shape=n_bins)

        # get bounds from the environment
        self.bounds = environment.gym_spaces_bounds

        # check valid values
        if not 0 < alpha < 1:
            warnings.warn("alpha should be in (0, 1)")
        if not 0 < gamma < 1:
            warnings.warn("gamma should be in (0, 1)")
        if not 0 <= epsilon_zero <= 1:
            warnings.warn("epsilon should be in [0, 1]")

        # save last action indexes for training
        self._last_action = None

    def bin_obs(self, obs: dict[str, float]) -> tuple[int, int]:
        """Discretize the continuous observation space

        The agent needs discrete values to index the Q table.
        The continous space is divided into n_bins for each feature.
        To do so, the observation is compared to poles that are linearly spaced between the bounds.
        The assigned index is the index of the closest pole.

        Args:
            obs: observation from the environment

        Returns:
            indices of the Q table corresponding to the observation

        """
        obs = np.array([obs["firm_stock"], obs["price_delta"]])
        firm_stock = obs[0]
        price_delta = obs[1]

        # linearly create bins
        firm_stock_poles = np.linspace(self.bounds["obs_firm_stock"][0],
                                       self.bounds["obs_firm_stock"][1],
                                       self.Q.shape[0])
        price_delta_poles = np.linspace(self.bounds["obs_price_delta"][0],
                                        self.bounds["obs_price_delta"][1],
                                        self.Q.shape[1])

        # Q matrix index is closest pole
        firm_stock_idx = np.abs(firm_stock_poles - firm_stock).argmin()
        price_delta_idx = np.abs(price_delta_poles - price_delta).argmin()

        return firm_stock_idx, price_delta_idx

    def get_action(self, obs: dict[str, float]) -> tuple[float, float]:
        """Choose an action based on epsilon-greedy policy

        The RL agent bins the observation obtained from the environment and chooses the action to take.
        The action is chosen at random with probability epsilon, and with probability 1-epsilon, the action is chosen
        based on the maximum Q value of Q(obs[0], obs[1], *, *).
        The action is then mapped to the continuous action space, linearly divided in bins between the environment
        provided bounds.

        Args:
            obs: observation from the environment

        Returns:
            chosen production factor and price factor

        """
        # get the Q-table indexes of the action
        if np.random.rand() < self.epsilon:
            prod_index = np.random.randint(0, self.Q.shape[2])
            price_index = np.random.randint(0, self.Q.shape[3])
        else:  # take action with max Q value
            firm_stock_idx, price_delta_idx = self.bin_obs(obs)
            prod_index, price_index = np.unravel_index(
                np.argmax(self.Q[firm_stock_idx, price_delta_idx]),
                self.Q.shape[2:]
            )
        self._last_action = (prod_index, price_index)

        # get the action values s.t. 0 goes to the lower bound and 1 goes to the upper bound
        prod_factor = self.bounds["act_production_factor"][0] + prod_index * (
                self.bounds["act_production_factor"][1] - self.bounds["act_production_factor"][0]) / (self.Q.shape[2]-1)
        price_factor = self.bounds["act_price_factor"][0] + price_index * (
                self.bounds["act_price_factor"][1] - self.bounds["act_price_factor"][0]) / (self.Q.shape[3]-1)

        return prod_factor, price_factor
