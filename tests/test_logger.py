import unittest
import shutil
from pathlib import Path

from rmabm import Logger, Simulation
from rmabm.environments import Cats
from rmabm.agents import Dummy, QLearner

import numpy as np


class LoggerTestCase(unittest.TestCase):

    def setUp(self):
        # create a directory for the test logs
        self.log_dir = Path("log_test_files")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # remove the test logs
        shutil.rmtree(self.log_dir)

    def test_class_init(self):
        # log dir as str
        log_directory = self.log_dir / "test_1"
        logger1 = Logger(log_name="init_test1", log_directory=str(log_directory), use_timestamp=False)
        self.assertIsInstance(logger1, Logger)
        self.assertTrue(logger1.path.is_dir())
        self.assertTrue(logger1.path.exists())
        self.assertEqual(logger1.path, log_directory / "init_test1")
        del logger1

        # log dir as Path
        log_directory = self.log_dir / "test_2"
        logger2 = Logger(log_name="init_test2", log_directory=log_directory, use_timestamp=False)
        self.assertIsInstance(logger2, Logger)
        self.assertTrue(logger2.path.is_dir())
        self.assertTrue(logger2.path.exists())
        self.assertEqual(logger2.path, log_directory / "init_test2")
        del logger2

        # with timestamp
        log_directory = self.log_dir / "test_3"
        logger3 = Logger(log_name="init_test3", log_directory=log_directory, use_timestamp=True)
        self.assertIsInstance(logger3, Logger)
        self.assertTrue(logger3.path.is_dir())
        self.assertTrue(logger3.path.exists())
        self.assertTrue(logger3.path.name.startswith("init_test3_"))
        del logger3

    def test_directory_creation(self):
        # existing directory
        logger1 = Logger(log_name="dir_test1", log_directory=self.log_dir, use_timestamp=False)
        self.assertTrue(logger1.path.is_dir())
        self.assertTrue(logger1.path.exists())
        self.assertEqual(logger1.path, self.log_dir / "dir_test1")
        # there should be only one directory in the log_dir
        self.assertEqual(len(list(self.log_dir.iterdir())), 1)
        del logger1

        # non-existing directory
        log_directory = self.log_dir / "subdirectory"
        logger2 = Logger(log_name="dir_test2", log_directory=log_directory, use_timestamp=False)
        self.assertTrue(logger2.path.is_dir())
        self.assertTrue(logger2.path.exists())
        self.assertEqual(logger2.path, log_directory / "dir_test2")
        # there should be two directories in the log_dir
        self.assertEqual(len(list(self.log_dir.iterdir())), 2)
        # the subdirectory should have 1 file
        self.assertEqual(len(list(log_directory.iterdir())), 1)
        del logger2

        # existing directory with timestamp
        log_directory = self.log_dir / "subdirectory"
        logger3 = Logger(log_name="dir_test3", log_directory=log_directory, use_timestamp=True)
        self.assertTrue(logger3.path.is_dir())
        self.assertTrue(logger3.path.exists())
        self.assertTrue(logger3.path.name.startswith("dir_test3_"))
        # there should be two directories in the log_dir
        self.assertEqual(len(list(self.log_dir.iterdir())), 2)
        # there should be two directories in log_dir/subdirectory
        self.assertEqual(len(list(log_directory.iterdir())), 2)
        del logger3

    def test_log_array(self):
        logger = Logger(log_name="array_test", log_directory=self.log_dir, use_timestamp=False)
        test_array = np.array([1, 2, 3, 4, 5]).reshape(1, 5)

        # new array
        logger.log_array(test_array, "test_array")
        self.assertTrue((logger.path / "test_array.npy").exists())
        self.assertTrue((self.log_dir / "array_test" / "test_array.npy").exists())
        self.assertEqual(len(list(logger.path.iterdir())), 1)

        self.assertEqual(np.load(logger.path / "test_array.npy").shape, test_array.shape)
        self.assertTrue((np.load(logger.path / "test_array.npy") == test_array).all())

        expected_array = test_array.copy()
        n = 5
        for _ in range(n-1):
            expected_array = np.concatenate((expected_array, test_array), axis=0)

            # append to existing array
            logger.log_array(test_array, "test_array")
            self.assertTrue((logger.path / "test_array.npy").exists())
            self.assertTrue((self.log_dir / "array_test" / "test_array.npy").exists())
            self.assertEqual(len(list(logger.path.iterdir())), 1)

        self.assertEqual(np.load(logger.path / "test_array.npy").shape, (n, 5))

    def test_log_info(self):
        logger = Logger(log_name="dict_test", log_directory=self.log_dir, use_timestamp=False)
        test_dict_list = [
            {"key1": 0, "key2": 1},
            {"key1": 3, "key2": 4},
            {"key1": 6, "key2": 7},
        ]
        key1_array = np.array([0, 3, 6])
        key2_array = np.array([1, 4, 7])

        logger.log_info(test_dict_list)
        self.assertTrue((logger.path / "key1.npy").exists())
        self.assertTrue((logger.path / "key2.npy").exists())
        self.assertTrue((self.log_dir / "dict_test" / "key1.npy").exists())
        self.assertTrue((self.log_dir / "dict_test" / "key2.npy").exists())

        # there should be two files in the log subdirectory and one in the main log dir
        self.assertEqual(len(list(logger.path.iterdir())), 2)
        self.assertEqual(len(list(self.log_dir.iterdir())), 1)

        # value checks
        self.assertEqual(np.load(logger.path / "key1.npy").shape, key1_array.shape)
        self.assertTrue((np.load(logger.path / "key1.npy") == key1_array).all())
        self.assertEqual(np.load(logger.path / "key2.npy").shape, key2_array.shape)
        self.assertTrue((np.load(logger.path / "key2.npy") == key2_array).all())


class SimulationLoggerTestCase(unittest.TestCase):

    def setUp(self):
        # create a directory for the test logs
        self.log_dir = Path("log_test_files")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_length = 10

    def tearDown(self):
        # remove the test logs
        shutil.rmtree(self.log_dir)

    def test_class_init(self):
        env = Cats()
        sim = Simulation(environment=env, agents=[])
        sim.init_logger(log_name="init_test", log_directory=self.log_dir, use_timestamp=False)

        self.assertIsInstance(sim.logger, Logger)

    def info_check(self, prefix):
        self.assertTrue((prefix / "ep1_Y_real.npy").exists())
        self.assertTrue((prefix / "ep1_wb.npy").exists())
        self.assertTrue((prefix / "ep1_Un.npy").exists())
        self.assertTrue((prefix / "ep1_bankruptcy_rate.npy").exists())
        self.assertTrue((prefix / "ep1_avg_price.npy").exists())

        # check the shape of the arrays
        expected_shape = (self.episode_length,)
        self.assertEqual(np.load(prefix / "ep1_Y_real.npy").shape, expected_shape)
        self.assertEqual(np.load(prefix / "ep1_wb.npy").shape, expected_shape)
        self.assertEqual(np.load(prefix / "ep1_Un.npy").shape, expected_shape)
        self.assertEqual(np.load(prefix / "ep1_bankruptcy_rate.npy").shape, expected_shape)
        self.assertEqual(np.load(prefix / "ep1_avg_price.npy").shape, expected_shape)

    def test_log_episode_no_agents(self):
        env = Cats(n_agents=0, T=self.episode_length, info_level=1)
        sim = Simulation(environment=env, agents=[])
        sim.init_logger(log_name="no_agents", log_directory=self.log_dir, use_timestamp=False)
        sim.run_episode()

        prefix = self.log_dir / "no_agents"

        # there should be only info files in the log directory
        self.info_check(prefix)
        self.assertFalse((prefix / "ep1_obs.npy").exists())
        self.assertFalse((prefix / "ep1_act.npy").exists())
        self.assertFalse((prefix / "ep1_rewards.npy").exists())

    def test_log_episode_dummy_agent(self):
        env = Cats(n_agents=1, T=self.episode_length, info_level=1)
        agent = Dummy(env.agents_ids[0])
        sim = Simulation(environment=env, agents=[agent])
        sim.init_logger(log_name="dummy_agent", log_directory=self.log_dir, use_timestamp=False)
        sim.run_episode()

        prefix = self.log_dir / "dummy_agent"

        self.info_check(prefix)
        self.assertTrue((prefix / "ep1_obs.npy").exists())
        self.assertTrue((prefix / "ep1_rewards.npy").exists())
        self.assertFalse((prefix / "ep1_act.npy").exists())  # only dummies are not logged

        # check the shape of the arrays
        self.assertEqual(np.load(prefix / "ep1_obs.npy").shape, (self.episode_length, 1, 2))
        self.assertEqual(np.load(prefix / "ep1_rewards.npy").shape, (self.episode_length, 1))

    def test_log_episode_qlearner_agent(self):
        env = Cats(n_agents=1, T=self.episode_length, info_level=1)
        agent = QLearner(env.agents_ids[0], env)
        sim = Simulation(environment=env, agents=[agent])
        sim.init_logger(log_name="ql_agent", log_directory=self.log_dir, use_timestamp=False)
        sim.run_episode()

        prefix = self.log_dir / "ql_agent"

        self.info_check(prefix)
        self.assertTrue((prefix / "ep1_obs.npy").exists())
        self.assertTrue((prefix / "ep1_act.npy").exists())
        self.assertTrue((prefix / "ep1_rewards.npy").exists())

        # check the shape of the arrays
        self.assertEqual(np.load(prefix / "ep1_obs.npy").shape, (self.episode_length, 1, 2))
        self.assertEqual(np.load(prefix / "ep1_act.npy").shape, (self.episode_length, 1, 2))
        self.assertEqual(np.load(prefix / "ep1_rewards.npy").shape, (self.episode_length, 1))

    def test_log_episode_multiple_agents(self):
        env = Cats(n_agents=2, T=self.episode_length, info_level=1)
        agents = [Dummy(env.agents_ids[0]), QLearner(env.agents_ids[1], env)]
        sim = Simulation(environment=env, agents=agents)
        sim.init_logger(log_name="multi_agents", log_directory=self.log_dir, use_timestamp=False)
        sim.run_episode()

        prefix = self.log_dir / "multi_agents"

        self.info_check(prefix)
        self.assertTrue((prefix / "ep1_obs.npy").exists())
        self.assertTrue((prefix / "ep1_act.npy").exists())
        self.assertTrue((prefix / "ep1_rewards.npy").exists())

        # check the shape of the arrays
        self.assertEqual(np.load(prefix / "ep1_obs.npy").shape, (self.episode_length, 2, 2))
        self.assertEqual(np.load(prefix / "ep1_act.npy").shape, (self.episode_length, 2, 2))
        self.assertEqual(np.load(prefix / "ep1_rewards.npy").shape, (self.episode_length, 2))


if __name__ == '__main__':
    unittest.main()
