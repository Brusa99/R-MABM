import unittest

from pycats.agents.qlearners import QLearner
from pycats.environments import Cats


class AgentTestCase(unittest.TestCase):
    agent_id = 0
    n_bins = 9
    alpha = 0.1
    gamma = 0.99
    epsilon_zero = 0.9
    eps_update = None

    bounds = {
        "obs_firm_stock": (-4.0, 4.0),
        "obs_price_delta": (-1.0, 7.0),
        "act_production_factor": (0.8, 1.2),
        "act_price_factor": (0.8, 1.2),
    }
    env = Cats(gym_spaces_bounds=bounds)

    def test_agent_init(self):
        agent = QLearner(
            agent_id=self.agent_id,
            environment=self.env,
            n_bins=self.n_bins,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon_zero=self.epsilon_zero,
        )
        self.assertEqual(agent.agent_id, self.agent_id)
        self.assertEqual(agent.alpha, self.alpha)
        self.assertEqual(agent.gamma, self.gamma)
        self.assertEqual(agent.epsilon, self.epsilon_zero)
        self.assertEqual(agent.Q.shape, (self.n_bins, self.n_bins, self.n_bins, self.n_bins))

    def test_bin_obs(self):
        agent = QLearner(
            agent_id=self.agent_id,
            environment=self.env,
            n_bins=self.n_bins,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon_zero=self.epsilon_zero,
        )

        # lower than bounds
        obs = {"firm_stock": -5.0, "price_delta": -2.0}
        obs_idx = agent.bin_obs(obs)
        self.assertEqual(obs_idx, (0, 0))

        # higher than bounds
        obs = {"firm_stock": 5.0, "price_delta": 8.0}
        obs_idx = agent.bin_obs(obs)
        self.assertEqual(obs_idx, (8, 8))

        # in the middle
        obs = {"firm_stock": 0.2, "price_delta": 0.2}
        obs_idx = agent.bin_obs(obs)
        self.assertEqual(obs_idx, (4, 1))

    def test_actions(self):
        agent = QLearner(
            agent_id=self.agent_id,
            environment=self.env,
            n_bins=self.n_bins,
            epsilon_zero=0,
        )

        # take an observation that will get mapped to (0, 0)
        obs = {"firm_stock": -100, "price_delta": -100}
        # max action indexes in Q table
        max_act_idx = (self.n_bins-1, self.n_bins-1)

        # increase Q value for max action in obs (0, 0)
        agent.Q[0, 0, *max_act_idx] = 1

        prod_factor, price_factor = agent.get_action(obs)
        # check indexes
        self.assertEqual(agent._last_action, max_act_idx)
        # check values
        self.assertEqual(prod_factor, 1.2)
        self.assertEqual(price_factor, 1.2)


if __name__ == '__main__':
    unittest.main()
