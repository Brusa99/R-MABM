import unittest
import shutil
from pathlib import Path

from rmabm import Simulation
from rmabm.agents import QLearner, Dummy
from rmabm.environments import Cats


class SimulationTestCase(unittest.TestCase):
    env = Cats(n_agents=2, T=500)
    ids = env.agents_ids
    agents = [QLearner(agent_id=ids[0], environment=env), Dummy(agent_id=ids[1])]

    n_episodes = 200
    current_episode = 2

    def test_init(self):
        sim = Simulation(
            environment=self.env,
            agents=self.agents,
            n_episodes=self.n_episodes,
            current_episode=self.current_episode
        )
        self.assertIsInstance(sim, Simulation)
        self.assertEqual(sim.env, self.env)

        # check agents
        self.assertEqual(sim.agents, self.agents)
        self.assertEqual(sim.qlearners, [self.agents[0]])
        self.assertEqual(sim.agents[0], self.agents[0])
        self.assertEqual(sim.agents[1], self.agents[1])

        self.assertEqual(sim.n_episodes, self.n_episodes)
        self.assertEqual(sim.current_episode, 2)
        self.assertIsNone(sim.logger)

        # check data storage
        self.assertEqual(sim._obs_history, [])
        self.assertEqual(sim._act_history, [])
        self.assertEqual(sim._reward_history, [])
        self.assertEqual(sim._info_history, [])

        # check initial state
        self.assertIsNotNone(sim._obs)
        self.assertIsNotNone(sim._info)
        self.assertIsNone(sim._next_obs)
        self.assertIsNone(sim._reward)
        self.assertFalse(sim._terminated)
        self.assertFalse(sim._truncated)

    def test_init_logger(self):
        sim = Simulation(
            environment=self.env,
            agents=self.agents,
            n_episodes=self.n_episodes,
            current_episode=self.current_episode
        )
        sim.init_logger(log_name="test_log", log_directory="logs", use_timestamp=False)
        self.assertIsNotNone(sim.logger)
        self.assertEqual(sim.logger._log_name, "test_log")
        self.assertEqual(sim.logger.path, Path("logs/test_log"))

        # remove the test logs
        shutil.rmtree("logs")

    def test_step(self):
        sim = Simulation(
            environment=self.env,
            agents=self.agents,
            n_episodes=self.n_episodes,
            current_episode=self.current_episode
        )
        # no training
        sim.step()
        self.assertIsNotNone(sim._obs)
        self.assertIsNotNone(sim._info)
        self.assertIsNotNone(sim._next_obs)
        self.assertIsNotNone(sim._reward)
        self.assertFalse(sim._terminated)
        self.assertFalse(sim._truncated)

        self.assertEqual(len(sim.obs_history), 1)
        self.assertEqual(len(sim.act_history), 1)
        self.assertEqual(len(sim.reward_history), 1)
        self.assertEqual(len(sim.info_history), 1)

        # training
        sim.step(train=True)
        self.assertIsNotNone(sim._obs)
        self.assertIsNotNone(sim._info)
        self.assertIsNotNone(sim._next_obs)
        self.assertIsNotNone(sim._reward)
        self.assertFalse(sim._terminated)
        self.assertFalse(sim._truncated)

        self.assertEqual(len(sim.obs_history), 2)
        self.assertEqual(len(sim.act_history), 2)
        self.assertEqual(len(sim.reward_history), 2)
        self.assertEqual(len(sim.info_history), 2)

        # check that the agent Q matrix has at least a non-zero value
        self.assertNotEqual(max(sim.agents[0].Q.max(), - sim.agents[0].Q.min()), 0)

    def test_reset(self):
        sim = Simulation(
            environment=self.env,
            agents=self.agents,
            n_episodes=self.n_episodes,
            current_episode=self.current_episode
        )

        sim.step()
        sim.reset()

        self.assertIsNotNone(sim._obs)
        self.assertIsNotNone(sim._info)
        self.assertIsNone(sim._next_obs)
        self.assertIsNone(sim._reward)
        self.assertFalse(sim._terminated)
        self.assertFalse(sim._truncated)

        self.assertEqual(sim._obs_history, [])
        self.assertEqual(sim._act_history, [])
        self.assertEqual(sim._reward_history, [])
        self.assertEqual(sim._info_history, [])

    def test_run_episode(self):
        sim = Simulation(
            environment=self.env,
            agents=self.agents,
            n_episodes=self.n_episodes,
            current_episode=self.current_episode
        )
        sim.run_episode()
        self.assertEqual(sim.current_episode, self.current_episode + 1)


if __name__ == '__main__':
    unittest.main()
