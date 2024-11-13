import unittest

from rmabm.agents import QLearner, Dummy
from rmabm.environments import Cats


class QLearnerTestCase(unittest.TestCase):
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

    def test_get_action(self):
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

    def test_train(self):
        agent = QLearner(
            agent_id=self.agent_id,
            environment=self.env,
            n_bins=3,
            epsilon_zero=0,
            gamma=self.gamma,
            alpha=self.alpha
        )
        # take an observation that will get mapped to (0, 0) and (2, 2)
        obs = {"firm_stock": -100, "price_delta": -100}
        next_obs = {"firm_stock": 100, "price_delta": 100}
        reward = 1

        # perform the step to get the action (we manually set the Q value to 1)
        agent.Q[0, 0, 0, 0] = 1
        agent.get_action(obs)  # this will set _last_action to (0, 0)

        # perform the train step
        agent.train(obs, reward, next_obs, terminated=False)
        # check the Q value
        exp_value = (1 - self.alpha) * 1 + self.alpha * (reward + self.gamma * 0)
        self.assertEqual(agent.Q[0, 0, 0, 0], exp_value)


class DummyTestCase(unittest.TestCase):
    agent_id = 0

    def test_agent_init(self):
        agent = Dummy(agent_id=self.agent_id)
        self.assertEqual(agent.agent_id, self.agent_id)
        self.assertEqual(agent.action_length, 2)

        env = Cats(n_agents=1)
        agent = Dummy(agent_id=env.agents_ids[0], environment=env)
        self.assertEqual(agent.agent_id, env.agents_ids[0])
        self.assertEqual(agent.action_length, 2)

    def test_bin_obs(self):
        agent = Dummy(agent_id=self.agent_id)
        self.assertEqual(agent.bin_obs(), (-1, -1))

    def test_get_action(self):
        agent = Dummy(agent_id=self.agent_id)
        self.assertEqual(agent.get_action(), ("dummy", "dummy"))

    def test_train(self):
        agent = Dummy(agent_id=self.agent_id)
        agent.train()


if __name__ == '__main__':
    unittest.main()
