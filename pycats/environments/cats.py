import random
from typing import Union, Dict, Optional, Tuple, Any
from pathlib import Path
import warnings

import numpy as np

import gymnasium as gym
from gymnasium.core import ObsType, RenderFrame

import juliacall
from juliacall import Main as jl


class Cats(gym.Env):
    """Implementation of the CATS model as an OpenAI Gym environment.

    The environment is implemented in a multi-agent setting. The original model presents agents that follow a given
    heuristic to set prices and production levels. In this implementation, a number of agents can be replaced by
    reinforcement learning agents that learn to set prices and production levels to maximize their rewards.

    The model parameters are controlled by the `T`, `W`, `F`, `N` and `params` args. The other parameters are used
    to control the RL agents learning and capabilites.

    Args:
        T: Number of steps in the simulation
        W: Number of workers
        F: Number of consumption-goods producing firms
        N: Number of capital-goods producing firms
        params: Parameters to be used bu the model.
        t_burnin: Number of steps performed before agents introduction.
        n_agents: Number of RL agents.
        reward_type: 'profits' or 'rms', the reward type for the agents. 'rms' is the revenue market share: given by
            the ratio of the agent's revenue to the total revenue in the market.
        price_change: 'agent_price' or 'avg_price', the price change mechanism for the agents. With 'agent_price',
            the agent's price is changed based on the previous step values. With 'avg_price', the agent's price is
            changed based on the average price of the market.
        bankruptcy_reward: The reward given to an agent when it goes bankrupt.
        render_mode: None, 'print', 'plot', 'save'. The render mode for the environment. If None, no rendering is
            done. If 'print' the environment info is printed at each step, if 'plot' the environment is plotted at
            each step, if 'save' the plots are both printed and saved.
        gym_spaces_bounds: Dictionary with the bounds for the observation and action spaces. The keys are:
            'obs_firm_stock', 'obs_price_delta', 'act_production_factor', 'act_price_factor'. The values are tuples with
            the lower and upper bounds for the spaces. If None, the default values are used. If the dictionary only has
            some of the keys, the default values are used for the missing keys.

    Attributes:
        t: current time step
        metadata: Metadata for the _gymnasium_ interface.
        
    """

    metadata: Dict = {
        "render.modes": ["print", "plot", "save"],
        "render_fps": 4,
    }

    _default_parameters: Dict = {
        "z_c": 5,  # no. of aplications in consumption good market
        "z_k": 2,  # no. of aplications in capital good market
        "z_e": 5,  # number of job applications
        "xi": 0.96,  # memory parameter human wealth
        "chi": 0.05,  # fraction of wealth devoted to consumption
        "q_adj": 0.9,  # quantity adjustment parameter
        "p_adj": 0.1,  # price adjustment parameter
        "mu": 1.2,  # bank's gross mark-up
        "eta": 0.03,  # capital depreciation
        "Iprob": 0.25,  # probability of investing
        "phi": 0.002,  # bank's leverage parameter
        "theta": 0.05,  # rate of debt reimbursment
        "delta": 0.5,  # memory parameter in the capital utilization rate
        "alpha": 0.66667,  # labour productivity
        "k": 0.33333,  # capital productivity
        "div": 0.2,  # share of dividends
        "barX": 0.85,  # desired capital utilization
        "inventory_depreciation": 0.3,  # rate at which capital firms' inventories depreciate
        "b1": -15,  # Parameters for risk evaluation by banks
        "b2": 13,
        "b_k1": -5,
        "b_k2": 5,
        "interest_rate": 0.01,
        "subsidy": 0,
        "maastricht": 0.0,
        "target_deficit": 0.01,
        "tax_rate": 0.0,
        "wage_update_up": 0.1,
        "wage_update_down": 0.01,
        "u_target": 0.1,
        "wb": 1.0,  # initial wage rate
        "tax_rate_d": 0.0,  # taxes on dividends
        "r_f": 0.01,  # general refinancing rate
    }

    _default_gym_spaces_bounds: Dict = {
        "obs_firm_stock": (-5.0, 5.0),
        "obs_price_delta": (-2.0, 8.0),
        "act_production_factor": (0.7, 1.3),
        "act_price_factor": (0.7, 1.3),
    }

    def __init__(
            self,
            T: int = 5000,
            W: int = 1000,
            F: int = 100,
            N: int = 20,
            params: Dict[str, Any] | str | Path | None = None,
            t_burnin: int = 300,
            n_agents: int = 0,
            reward_type: str = "profits",
            price_change: str = "agent_price",
            bankruptcy_reward: int = -100,
            render_mode: Dict[str, Any] | None = None,
            gym_spaces_bounds: Dict[str, Tuple[float, float]] | None = None,
    ):
        self.T = T
        self.W = W
        self.F = F
        self.N = N
        self.t_burnin = t_burnin
        self.n_agents = n_agents
        self.bankruptcy_reward = bankruptcy_reward

        self._load_parameters(params)

        # check valid arguments
        if reward_type not in ["profits", "rms"]:
            raise ValueError("Invalid reward type. Must be 'profits' or 'rms'")
        else:
            self.reward_type = reward_type

        if price_change not in ["agent_price", "avg_price"]:
            raise ValueError("Invalid price change mechanism. Must be 'agent_price' or 'avg_price'")
        else:
            self.price_change = price_change

        if render_mode not in self.metadata["render.modes"] and render_mode is not None:
            raise ValueError(f"Invalid render mode. Must be one of {self.metadata['render.modes']}")
        else:
            self.render_mode = render_mode

        # initialize the model through Julia
        jl.seval("using Cats")
        jl.seval("using Random")
        self._julia_model_init()

        # get the ids of RL agents and model-controlled agents
        _, ids_prod_firms, _, _, _ = jl.seval("Cats.get_ids")(self.model)
        self.agents_ids = ids_prod_firms[:n_agents]
        self.other_firms_ids = ids_prod_firms[n_agents:]  # TODO: refactor other_firms_ids to a more descriptive name

        self._create_spaces(gym_spaces_bounds)

        # Current time step
        self.t: int = 0

    def _load_parameters(self, params):
        """Load the parameters for the model. Automatically infers `param` type and loads the parameters accordingly."""

        if params is None:  # use default parameters
            self.params = juliacall.convert(T=jl.seval("Dict{Symbol, Real}"), x=self._default_parameters)
        # TODO: implement loading parameters from a file
        # TODO: implement loading parameters from a dictionary

    def _julia_model_init(self):
        """Initialize the model in Julia."""

        self.model = jl.seval("Cats.initialise_model")(self.W, self.F, self.N, self.params)
        self.model.bank["E_threshold"] = 30.0 * (self.F + self.N) * 0.1

    def _create_spaces(self, gym_spaces_bounds):
        """Create the gym spaces dicts for observations and actions"""

        # TODO: add check for key presence and add partial fill-in

        self.observation_space = gym.spaces.Dict(
            {
                "firm_stock": gym.spaces.Box(low=gym_spaces_bounds["obs_firm_stock"][0],
                                             high=gym_spaces_bounds["obs_firm_stock"][1],
                                             ),
                "price_delta": gym.spaces.Box(low=gym_spaces_bounds["obs_price_delta"][0],
                                              high=gym_spaces_bounds["obs_price_delta"][1],
                                              ),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                "production_factor": gym.spaces.Box(low=gym_spaces_bounds["act_production_factor"][0],
                                                    high=gym_spaces_bounds["act_production_factor"][1],
                                                    ),
                "price_factor": gym.spaces.Box(low=gym_spaces_bounds["act_price_factor"][0],
                                               high=gym_spaces_bounds["act_price_factor"][1],
                                               ),
            }
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def reset(
            self,
            seed: int | None = None,
            options: Dict[str, Any] | None = None,
    ) -> tuple[list[ObsType], dict[str, Any]]:
        """Reset the environment to a random initial state. Obtaining initial obs and info.

        To reset the environment, a new model is created through Julia.
        Burn-in is performed and each agents' observation are returned along with environment info.
        The seed is set across both python and julia.

        Args:
            seed:
            options:

        Returns:
            List of observations for each RL agents
            Environment info dictionary

        """
        # set seeds across everything
        super().reset(seed=seed)
        random.seed(seed)
        jl.seval("Random.seed!")(seed)
        np.random.seed(seed)

        # reinitialize the model
        self._julia_model_init()

        # burn-in the environment
        self._burnin()

        return self._get_obs(), self._get_info()