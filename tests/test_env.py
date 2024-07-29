import unittest
from pathlib import Path
import os

import gymnasium as gym
from rmabm.environments import Cats, CatsLog


class CatsTestCase(unittest.TestCase):
    T = 2000
    W = 800
    F = 90
    N = 15
    t_burnin = 100
    n_agents = 3
    bankruptcy_reward = -300
    info_level = 3

    def setUp(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def test_class_init(self):
        env = Cats(T=self.T, W=self.W, F=self.F, N=self.N, t_burnin=self.t_burnin, n_agents=self.n_agents,
                   bankruptcy_reward=self.bankruptcy_reward, info_level=self.info_level)

        self.assertIsInstance(env, Cats)
        self.assertEqual(env.T, self.T)
        self.assertEqual(env.W, self.W)
        self.assertEqual(env.F, self.F)
        self.assertEqual(env.N, self.N)
        self.assertEqual(env.t_burnin, self.t_burnin)
        self.assertEqual(env.n_agents, self.n_agents)
        self.assertEqual(env.bankruptcy_reward, self.bankruptcy_reward)
        self.assertEqual(env.info_level, self.info_level)

    def test_make(self):
        env = gym.make("Cats", T=self.T, W=self.W, F=self.F, N=self.N, t_burnin=self.t_burnin,
                       n_agents=self.n_agents, bankruptcy_reward=self.bankruptcy_reward, info_level=self.info_level)

        self.assertIsInstance(env.unwrapped, Cats)
        self.assertEqual(env.get_wrapper_attr('T'), self.T)
        self.assertEqual(env.get_wrapper_attr('W'), self.W)
        self.assertEqual(env.get_wrapper_attr('F'), self.F)
        self.assertEqual(env.get_wrapper_attr('N'), self.N)
        self.assertEqual(env.get_wrapper_attr('t_burnin'), self.t_burnin)
        self.assertEqual(env.get_wrapper_attr('n_agents'), self.n_agents)
        self.assertEqual(env.get_wrapper_attr('bankruptcy_reward'), self.bankruptcy_reward)
        self.assertEqual(env.get_wrapper_attr('info_level'), self.info_level)

    def test_load_params_from_full_dict(self):
        params = {
            "z_c": 6,  # no. of aplications in consumption good market
            "z_k": 1,  # no. of aplications in capital good market
            "z_e": 4,  # number of job applications
            "xi": 0.9,  # memory parameter human wealth
            "chi": 0.08,  # fraction of wealth devoted to consumption
            "q_adj": 0.8,  # quantity adjustment parameter
            "p_adj": 0.2,  # price adjustment parameter
            "mu": 1.1,  # bank's gross mark-up
            "eta": 0.04,  # capital depreciation
            "Iprob": 0.2,  # probability of investing
            "phi": 0.001,  # bank's leverage parameter
            "theta": 0.06,  # rate of debt reimbursment
            "delta": 0.4,  # memory parameter in the capital utilization rate
            "alpha": 0.7,  # labour productivity
            "k": 0.4,  # capital productivity
            "div": 0.3,  # share of dividends
            "barX": 0.8,  # desired capital utilization
            "inventory_depreciation": 0.2,  # rate at which capital firms' inventories depreciate
            "b1": -10,  # Parameters for risk evaluation by banks
            "b2": 10,
            "b_k1": -6,
            "b_k2": 6,
            "interest_rate": 0.02,
            "subsidy": 0.1,
            "maastricht": 0.02,
            "target_deficit": 0.02,
            "tax_rate": 0.1,
            "wage_update_up": 0.2,
            "wage_update_down": 0.02,
            "u_target": 0.2,
            "wb": 2.0,  # initial wage rate
            "tax_rate_d": 0.1,  # taxes on dividends
            "r_f": 0.02,  # general refinancing rate
        }
        env = Cats(params=params)

        self.assertEqual(env.params["z_c"], 6)
        self.assertEqual(env.params["z_k"], 1)
        self.assertEqual(env.params["z_e"], 4)
        self.assertEqual(env.params["xi"], 0.9)
        self.assertEqual(env.params["chi"], 0.08)
        self.assertEqual(env.params["q_adj"], 0.8)
        self.assertEqual(env.params["p_adj"], 0.2)
        self.assertEqual(env.params["mu"], 1.1)
        self.assertEqual(env.params["eta"], 0.04)
        self.assertEqual(env.params["Iprob"], 0.2)
        self.assertEqual(env.params["phi"], 0.001)
        self.assertEqual(env.params["theta"], 0.06)
        self.assertEqual(env.params["delta"], 0.4)
        self.assertEqual(env.params["alpha"], 0.7)
        self.assertEqual(env.params["k"], 0.4)
        self.assertEqual(env.params["div"], 0.3)
        self.assertEqual(env.params["barX"], 0.8)
        self.assertEqual(env.params["inventory_depreciation"], 0.2)
        self.assertEqual(env.params["b1"], -10)
        self.assertEqual(env.params["b2"], 10)
        self.assertEqual(env.params["b_k1"], -6)
        self.assertEqual(env.params["b_k2"], 6)
        self.assertEqual(env.params["interest_rate"], 0.02)
        self.assertEqual(env.params["subsidy"], 0.1)
        self.assertEqual(env.params["maastricht"], 0.02)
        self.assertEqual(env.params["target_deficit"], 0.02)
        self.assertEqual(env.params["tax_rate"], 0.1)
        self.assertEqual(env.params["wage_update_up"], 0.2)
        self.assertEqual(env.params["wage_update_down"], 0.02)
        self.assertEqual(env.params["u_target"], 0.2)
        self.assertEqual(env.params["wb"], 2.0)
        self.assertEqual(env.params["tax_rate_d"], 0.1)
        self.assertEqual(env.params["r_f"], 0.02)

    def test_load_params_from_partial_dict(self):
        params = {"mu": 1.1}
        def_params = Cats._default_parameters
        env = Cats(params=params)

        expected_params = params
        for key, value in def_params.items():
            if key not in expected_params:
                expected_params[key] = value

        self.assertEqual(env.params["mu"], 1.1)
        self.assertEqual(env.params["z_c"], expected_params["z_c"])

    def test_load_params_from_json(self):
        # path as string
        env = Cats(params="resources/test_parameters.json")
        self.assertEqual(env.params["z_c"], 4)
        self.assertEqual(env.params["z_k"], 2)  # checks if  the default value is used if not in the file

        # path as Path object
        env = Cats(params=Path("resources/test_parameters.json"))
        self.assertEqual(env.params["z_c"], 4)
        self.assertEqual(env.params["z_k"], 2)

    def test_load_params_from_csv(self):
        # path as string
        env = Cats(params="resources/test_parameters.csv")
        self.assertEqual(env.params["z_c"], 4)

        # path as Path object
        env = Cats(params=Path("resources/test_parameters.csv"))
        self.assertEqual(env.params["z_c"], 4)

    def test_step(self):
        env = Cats(T=self.T, W=self.W, F=self.F, N=self.N, t_burnin=self.t_burnin, n_agents=self.n_agents,
                   bankruptcy_reward=self.bankruptcy_reward, info_level=self.info_level)
        env.reset()
        sample_action = [[0, 0] for _ in range(self.n_agents)]

        # profit reward
        obs, reward, terminated, truncated, info = env.step(sample_action)
        self.assertEqual(len(obs), self.n_agents)
        self.assertEqual(len(reward), self.n_agents)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

        # rms reward
        env.reward_type = "rms"
        obs, reward, terminated, truncated, info = env.step(sample_action)
        self.assertEqual(len(obs), self.n_agents)
        self.assertEqual(len(reward), self.n_agents)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)



class CatsLogTestCase(unittest.TestCase):
    T = 2000
    W = 800
    F = 90
    N = 15
    t_burnin = 100
    n_agents = 3
    bankruptcy_reward = -300

    def test_class_init(self):
        env = CatsLog(T=self.T, W=self.W, F=self.F, N=self.N, t_burnin=self.t_burnin, n_agents=self.n_agents,
                      bankruptcy_reward=self.bankruptcy_reward)

        self.assertIsInstance(env, CatsLog)
        self.assertIsInstance(env, Cats)
        self.assertEqual(env.T, self.T)
        self.assertEqual(env.W, self.W)
        self.assertEqual(env.F, self.F)
        self.assertEqual(env.N, self.N)
        self.assertEqual(env.t_burnin, self.t_burnin)
        self.assertEqual(env.n_agents, self.n_agents)
        self.assertEqual(env.bankruptcy_reward, self.bankruptcy_reward)

    def test_make(self):
        env = gym.make("CatsLog", T=self.T, W=self.W, F=self.F, N=self.N, t_burnin=self.t_burnin,
                       n_agents=self.n_agents, bankruptcy_reward=self.bankruptcy_reward)

        self.assertIsInstance(env.unwrapped, CatsLog)
        self.assertIsInstance(env.unwrapped, Cats)
        self.assertEqual(env.get_wrapper_attr('T'), self.T)
        self.assertEqual(env.get_wrapper_attr('W'), self.W)
        self.assertEqual(env.get_wrapper_attr('F'), self.F)
        self.assertEqual(env.get_wrapper_attr('N'), self.N)
        self.assertEqual(env.get_wrapper_attr('t_burnin'), self.t_burnin)
        self.assertEqual(env.get_wrapper_attr('n_agents'), self.n_agents)
        self.assertEqual(env.get_wrapper_attr('bankruptcy_reward'), self.bankruptcy_reward)

    def test_step(self):
        env = CatsLog(T=self.T, W=self.W, F=self.F, N=self.N, t_burnin=self.t_burnin, n_agents=self.n_agents,
                      bankruptcy_reward=self.bankruptcy_reward)
        env.reset()
        sample_action = [[0, 0] for _ in range(self.n_agents)]

        # profit reward
        obs, reward, terminated, truncated, info = env.step(sample_action)
        self.assertEqual(len(obs), self.n_agents)
        self.assertEqual(len(reward), self.n_agents)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

        # rms reward
        env.reward_type = "rms"
        obs, reward, terminated, truncated, info = env.step(sample_action)
        self.assertEqual(len(obs), self.n_agents)
        self.assertEqual(len(reward), self.n_agents)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)


if __name__ == '__main__':
    unittest.main()
