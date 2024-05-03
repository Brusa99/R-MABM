import unittest
import gymnasium as gym
import pycats


class ModelCreationTestCase(unittest.TestCase):
    T = 2000
    W = 800
    F = 90
    N = 15
    t_burnin = 100
    n_agents = 3
    bankruptcy_reward = -300

    def test_class_init(self):
        env = pycats.Cats(T=self.T, W=self.W, F=self.F, N=self.N, t_burnin=self.t_burnin, n_agents=self.n_agents,
                          bankruptcy_reward=self.bankruptcy_reward)

        self.assertIsInstance(env, pycats.Cats)
        self.assertEqual(env.T, self.T)
        self.assertEqual(env.W, self.W)
        self.assertEqual(env.F, self.F)
        self.assertEqual(env.N, self.N)
        self.assertEqual(env.t_burnin, self.t_burnin)
        self.assertEqual(env.n_agents, self.n_agents)
        self.assertEqual(env.bankruptcy_reward, self.bankruptcy_reward)

    def test_make(self):
        env = gym.make("Cats", T=self.T, W=self.W, F=self.F, N=self.N, t_burnin=self.t_burnin,
                       n_agents=self.n_agents, bankruptcy_reward=self.bankruptcy_reward)

        self.assertIsInstance(env.unwrapped, pycats.Cats)
        self.assertEqual(env.get_wrapper_attr('T'), self.T)
        self.assertEqual(env.get_wrapper_attr('W'), self.W)
        self.assertEqual(env.get_wrapper_attr('F'), self.F)
        self.assertEqual(env.get_wrapper_attr('N'), self.N)
        self.assertEqual(env.get_wrapper_attr('t_burnin'), self.t_burnin)
        self.assertEqual(env.get_wrapper_attr('n_agents'), self.n_agents)
        self.assertEqual(env.get_wrapper_attr('bankruptcy_reward'), self.bankruptcy_reward)


if __name__ == '__main__':
    unittest.main()
