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
