import random
import json
import csv
from typing import Dict, Tuple, Any, SupportsFloat
from typing_extensions import override
from pathlib import Path
import warnings
import math

import numpy as np

import gymnasium as gym
from gymnasium.core import ObsType, RenderFrame, ActType

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

    Raises:
        ValueError: If the reward_type, price_change or render_mode are invalid.
        
    """

    metadata: Dict = {
        "render_modes": ["print", "plot", "save"],
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
            self._price_change = price_change

        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
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

    def _load_parameters(self, params) -> None:
        """Load the parameters for the model. Automatically infers `param` type and loads the parameters accordingly."""

        if params is None:  # use default parameters
            self.params = juliacall.convert(T=jl.seval("Dict{Symbol, Real}"), x=self._default_parameters)
        elif isinstance(params, dict):
            # check for missing keys and fill them with default values
            for key, value in self._default_parameters.items():
                if key not in params:
                    params[key] = value
            self.params = juliacall.convert(T=jl.seval("Dict{Symbol, Real}"), x=params)

        if isinstance(params, str):
            params = Path(params)

        if isinstance(params, Path):
            if params.suffix == ".json":
                with open(params, "r") as f:
                    params = json.load(f)

                # check for missing keys and fill them with default values
                for key, value in self._default_parameters.items():
                    if key not in params:
                        params[key] = value

            elif params.suffix == ".csv":
                with open(params, "r") as f:
                    reader = csv.reader(f)
                    params = [float(row[0]) for row in reader]

                # first three parameters must be integers
                params[:3] = [int(x) for x in params[:3]]

                keys = list(self._default_parameters.keys())
                # check if the number of parameters is correct
                if len(params) != len(keys):
                    raise ValueError(f"Invalid number of parameters. Expected {len(keys)}, got {len(params)}")

                # create the dictionary
                params = {key: value for key, value in zip(keys, params)}

            else:
                raise ValueError(f"Invalid file type. Must be .json or .csv. Got {params.suffix}")

            self.params = juliacall.convert(T=jl.seval("Dict{Symbol, Real}"), x=params)

    def _julia_model_init(self) -> None:
        """Initialize the model in Julia."""

        self.model = jl.seval("Cats.initialise_model")(self.W, self.F, self.N, self.params)
        self.model.bank["E_threshold"] = 30.0 * (self.F + self.N) * 0.1

    def _create_spaces(self, gym_spaces_bounds) -> None:
        """Create the gym spaces dicts for observations and actions"""

        # use defaults if not provided
        if gym_spaces_bounds is None:
            gym_spaces_bounds = self._default_gym_spaces_bounds

        # check is all keys are present, fill with default values if not
        for key, value in self._default_gym_spaces_bounds.items():
            if key not in gym_spaces_bounds:
                gym_spaces_bounds[key] = value

        self.gym_spaces_bounds = gym_spaces_bounds

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
            seed: seed for the environment
            options: additional options for the reset, none are available for now

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

    def step(
        self,
        actions: list[ActType],
        burnin: bool = False,
    ) -> tuple[list[ObsType], list[SupportsFloat], bool, bool, dict[str, Any]]:
        """Perform a step in the environment.

        The agents' actions are applied to the model and the model is stepped forward.
        Agents' observations are returned along with their rewards. Moreover, termination and truncation flag are
        returned. The environment info is also returned.

        Termination occurs when all firms are bankrupt. Truncaion occurs when the time step reaches the maximum number
        of steps (`T`).

        Args:
            actions: list of actions for each agent. Each action is a dictionary with keys 'production_factor' and
                'price_factor'. Must be in the same order as the agents' ids.
            burnin: if True, provided actions are ignored and the model is stepped forward with original (rationally
                bounded) actions.

        Returns:
            List of observations for each RL agent
            List of rewards for each RL agent
            Termination flag
            Truncation flag
            Environment info dictionary

        Raises:
            ValueError: If the number of actions is different from the number of agents.

        """
        # check if input is valid
        if len(actions) != self.n_agents:
            raise ValueError(
                f"Number of actions ({len(actions)}) must be equal to the number of agents ({self.n_agents})"
            )
        # update the julia model from python
        ids_workers, ids_prod_firms, ids_cap_firms, ids_firms, ids = jl.seval("Cats.get_ids")(self.model)
        ids_workers_rand = ids_workers.copy()
        ids_prod_firms_rand = ids_prod_firms.copy()

        # simulation stops if all firms are bankrupt
        terminated = self._get_terminated(ids_prod_firms)

        jl.seval("Cats.reset_gov_and_banking_variables!")(self.model)
        self.model.gov["bond_interest_rate"] = self.model.params["interest_rate"]  # done through a function in julia

        # keep old values to make actions relative
        prev_demands = [self.model[f_id].De for f_id in self.agents_ids]
        prev_prices = [self.model[f_id].P for f_id in self.agents_ids]

        # firms decisions in the original mode
        jl.seval("Cats.find_expected_demand!")(ids_firms, self.model)
        jl.seval("Cats.find_expected_investment!")(ids_prod_firms, self.model)

        # overwrite the decision of the agents
        if not burnin:
            self._implement_action(actions, prev_demands, prev_prices)

        # model step
        jl.seval("Cats.find_labour_demand!")(ids_firms, self.model)
        jl.seval("Cats.get_credit!")(ids_firms, self.model)
        jl.seval("Cats.fire_workers!")(ids_firms, self.model, ids_workers_rand)
        jl.seval("Cats.search_job!")(ids_workers_rand, ids_firms, self.model)
        jl.seval("Cats.produce!")(ids_firms, self.model)
        jl.seval("Cats.buy_capgoods!")(ids_prod_firms, ids_cap_firms, self.model, ids_prod_firms_rand)
        jl.seval("Cats.adjust_subsidy!")(ids_workers, self.model)
        jl.seval("Cats.get_paid!")(ids_workers, self.model)
        jl.seval("Cats.find_cons_budget!")(ids, self.model)
        jl.seval("Cats.consume!")(ids, ids_prod_firms, self.model)

        # accounting and rewards
        jl.seval("Cats.accounting!")(ids_firms, self.model)
        reward = self._get_reward()

        # update avg prices and tracking variables
        jl.seval("Cats.update_price!")(self.model, ids_prod_firms)
        jl.seval("Cats.update_price_k!")(self.model, ids_cap_firms)
        jl.seval("Cats.update_tracking_variables!")(self.model, ids_prod_firms, ids_cap_firms, ids_workers)

        # check for bankruptcies and update gov and wages
        jl.seval("Cats.bankrupt!")(ids_prod_firms, ids_cap_firms, self.model, ids_workers)
        jl.seval("Cats.bank_pays_dividends!")(self.model, ids_prod_firms, ids_cap_firms)
        jl.seval("Cats.gov_accounting!")(self.model)
        jl.seval("Cats.adjust_wages!")(self.model)

        # update the time step
        self.model.timestep += 1
        self.t += 1

        # compute the state, reward and termination
        obs = self._get_obs()
        info = self._get_info()
        truncated = self._get_truncated()

        return obs, reward, terminated, truncated, info

    def _burnin(self) -> None:
        """Burn-in the environment. Resets class internal time to 0, julia model time is kept."""

        dummy_action = [[1, 1, 1] for _ in range(self.n_agents)]  # will be ignored by `step`

        for _ in range(self.t_burnin):
            self.step(dummy_action, burnin=True)

        # reset the time step
        self.t = 0

    def _implement_action(self, actions: list[ActType], prev_demands: list[float], prev_prices: list[float]) -> None:
        """Overwrite the decision of the julia agents with the provided actions for RL controlled firms."""

        for index, f_id in enumerate(self.agents_ids):
            # ignore dummy actions
            if actions[index] == "dummy":
                continue

            # unpack
            exp_demand_factor, firm_price_factor = actions[index]

            # compute the new values
            exp_demand = prev_demands[index] * exp_demand_factor

            if self._price_change == "agent_price":
                firm_price = prev_prices[index] * firm_price_factor
            elif self._price_change == "avg_price":
                firm_price = self.model.price * firm_price_factor
            else:
                raise ValueError(
                    f"Invalid price change mechanism. Must be 'agent_price' or 'avg_price'. Got {self._price_change} "
                    f"This should not happen, as it is checked in the __init__ method."
                )

            # adjust (minium demand is alpha = at least 1 worker)
            if exp_demand < self.model.params["alpha"]:
                exp_demand = self.model.params["alpha"]

            # modify in the julia model
            self.model[f_id].De = exp_demand
            self.model[f_id].P = firm_price

    def _get_obs(self) -> list[ObsType]:
        """Return the observations for the RL agents."""

        obs = []  # list of observations for all agents
        for f_id in self.agents_ids:
            # compute observations
            firm_stock = self.model[f_id].Y_prev - self.model[f_id].Yd
            price = self.model[f_id].P - self.model.price

            obs.append(
                {
                    "firm_stock": firm_stock,  # production surplus if > 0, additional sellable stock if > 0
                    "price_delta": price,  # price difference from the market average
                }
            )
        return obs

    def _get_terminated(self, ids_prod_firms: list[int]) -> bool:
        """Check if all firms are bankrupt."""

        return sum([self.model[f_id].A for f_id in ids_prod_firms]) == 0

    def _get_truncated(self) -> bool:
        """Check if the simulation reached the maximum number of steps."""

        return self.t >= self.T

    def _get_reward(self):
        """Compute the reward for the RL agents."""

        if self.reward_type == "profits":
            return self._get_profits_reward()
        elif self.reward_type == "rms":
            return self._get_rms_reward()
        else:
            raise ValueError(
                f"Invalid reward type. Must be 'profits' or 'rms'. Got {self.reward_type}. "
                "This should not happen, as it is checked in the __init__ method."
            )

    def _get_profits_reward(self) -> list[float]:
        """Reward is equal to profits unless the firm is bankrupt, in which case it is set to a negative value."""

        reward = []
        for f_id in self.agents_ids:
            # capital depreciation
            dep = self.model.params["eta"] * self.model[f_id].Y_prev / self.model.params["k"]
            try:
                dep_value_denominator = self.model[f_id].K + dep - self.model[f_id].investment
                dep_value = dep * self.model[f_id].capital_value / dep_value_denominator
            except ZeroDivisionError:
                dep_value = 0
                warnings.warn(
                    f"""division by zero in 'depreciation value' for agent {f_id}, setting to 0.
                    This is not expected and should be investigated.
                    'depreciation value' denominator is given by:
                        capital {self.model[f_id].K} + depreciation {dep} - investment {self.model[f_id].investment}."""
                )

            # revenues
            RIC = self.model[f_id].P * self.model[f_id].Q

            # calculate profits and transform them in real terms
            profits = RIC - self.model[f_id].wages - self.model[f_id].interests - dep_value
            profits = profits * self.model.init_price / self.model.price

            # nan values are set to 0 (underflow)
            if np.isnan(profits):
                profits = 0
                warnings.warn(f"profits for agent {f_id} are nan (likely underflow), setting to 0.")

            # overwrite with a substantial negative reward if the firm is bankrupt
            if self.model[f_id].A <= 0:
                profits = self.bankruptcy_reward

            reward.append(profits)
        return reward

    def _get_rms_reward(self) -> list[float]:
        """Reward is equal to the revenue market share of the agent."""

        # calculate revenue of all C-firms
        total_revenue = sum([self.model[f_id].P * self.model[f_id].Q for f_id in self.agents_ids])
        total_revenue += sum([self.model[f_id].P * self.model[f_id].Q for f_id in self.other_firms_ids])

        # calculate the reward for each agent
        reward = []
        for f_id in self.agents_ids:
            RIC = self.model[f_id].P * self.model[f_id].Q
            reward.append(RIC / total_revenue)
        return reward

    def _get_info(self) -> dict[str, Any]:
        """Return the environment info."""

        # TODO: add info level variable to control the amount of information returned

        agents_production = [self.model[agent_id].Y_prev * self.model.price for agent_id in self.agents_ids]
        others_production = np.array([self.model[f_id].Y_prev * self.model.price for f_id in self.other_firms_ids])
        agents_sales = [self.model[agent_id].Q for agent_id in self.agents_ids]
        others_sales = np.array([self.model[f_id].Q for f_id in self.other_firms_ids])
        agents_debt = [self.model[agent_id].deb for agent_id in self.agents_ids]
        others_debt = np.array([self.model[f_id].deb for f_id in self.other_firms_ids])
        agents_employed = [self.model[agent_id].Leff for agent_id in self.agents_ids]
        others_employed = np.array([self.model[f_id].Leff for f_id in self.other_firms_ids])
        agents_capital = [self.model[agent_id].K for agent_id in self.agents_ids]
        others_capital = np.array([self.model[f_id].K for f_id in self.other_firms_ids])
        agents_equity = [self.model[agent_id].A for agent_id in self.agents_ids]
        others_equity = np.array([self.model[f_id].A for f_id in self.other_firms_ids])
        agents_investment = [self.model[agent_id].investment for agent_id in self.agents_ids]
        others_investment = np.array([self.model[f_id].investment for f_id in self.other_firms_ids])
        agents_liquidity = [self.model[agent_id].liquidity for agent_id in self.agents_ids]
        others_liquidity = np.array([self.model[f_id].liquidity for f_id in self.other_firms_ids])

        info = {
            # model variables
            "Y_real": self.model.Y_real,                    # GDP adjusted for inflation
            "Y_nominal_tot": self.model.Y_nominal_tot,      # nominal GDP
            "gdp_deflator": self.model.gdp_deflator,
            "inflationRate": self.model.inflationRate,
            "consumption": self.model.consumption,
            "wb": self.model.wb,                            # wage rate
            "Un": self.model.Un,                            # unemployment rate
            "bankruptcy_rate": self.model.bankruptcy_rate,
            "totalDeb": self.model.totalDeb,
            "totalDeb_k": self.model.totalDeb_k,
            "Investment": self.model.Investment,
            "totK": self.model.totK,
            "inventories": self.model.inventories,
            "totE": self.model.totE,
            "dividendsB": self.model.dividendsB,
            "profitsB": self.model.profitsB,
            "GB": self.model.GB,
            "deficitGDP": self.model.deficitGDP,
            "bonds": self.model.bonds,
            "avg_price": self.model.price,                  # average production good price

            # firms variables
            "agents_production": agents_production,
            "others_production": (np.mean(others_production), np.std(others_production)),
            "agents_sales": agents_sales,
            "others_sales": (np.mean(others_sales), np.std(others_sales)),
            "agents_debt": agents_debt,
            "others_debt": (np.mean(others_debt), np.std(others_debt)),
            "agents_employment": agents_employed,
            "others_employment": (np.mean(others_employed), np.std(others_employed)),
            "agents_capital": agents_capital,
            "others_capital": (np.mean(others_capital), np.std(others_capital)),
            "agents_equity": agents_equity,
            "others_equity": (np.mean(others_equity), np.std(others_equity)),
            "agents_investment": agents_investment,
            "others_investment": (np.mean(others_investment), np.std(others_investment)),
            "agents_liquidity": agents_liquidity,
            "others_liquidity": (np.mean(others_liquidity), np.std(others_liquidity)),
        }
        return info


class CatsLog(Cats):
    """Cats environment with logarithmic observations and actions.

    This class overrides the `Cats` environment to provide a version with logarithmic observations and actions.
    In particular the differences are:

    * Observations:
        - `firm_stock` is given by log(production/demand)
        - `price_delta` is given by log(agent_price/avg_price)
    * Actions:
        - `production_factor` is given by exp(log(demand) + action_value[0])
        - `price_factor` is given by exp(log(seleced_price) + action_value[1]), where _selected_price_ depends on the
          price mechanism selected (agent_price or avg_price).

    See Also:
        Cats: The original Cats environment.

    """

    _default_gym_spaces_bounds: Dict = {
        "obs_firm_stock": (-2.0, 2.0),
        "obs_price_delta": (-2.0, 2.0),
        "act_production_factor": (-0.3, 0.3),
        "act_price_factor": (-0.3, 0.3),
    }

    @override
    def _get_obs(self) -> list[ObsType]:
        """Return the (logaritmic version of the) observations for the RL agents."""

        obs = []
        for f_id in self.agents_ids:
            # compute observations
            firm_stock = math.log(self.model[f_id].Y_prev + 1e-8) - math.log(self.model[f_id].Yd + 1e-8)
            price = math.log(self.model[f_id].P + 1e-8) - math.log(self.model.price + 1e-8)

            obs.append(
                {
                    "firm_stock": firm_stock,
                    "price_delta": price,
                }
            )
        return obs

    @override
    def _implement_action(self, actions: list[ActType], prev_demands: list[float], prev_prices: list[float]) -> None:
        """Overwrite the decision of the julia agents with the provided actions for RL controlled firms."""

        for index, f_id in enumerate(self.agents_ids):
            # ignore dummy actions
            if actions[index] == "dummy":
                continue

            # unpack
            exp_demand_factor, firm_price_factor = actions[index]

            # compute the new values
            exp_demand = math.exp(math.log(prev_demands[index] + 1e-8) + exp_demand_factor)

            if self._price_change == "agent_price":
                firm_price = math.exp(math.log(prev_prices[index] + 1e-8) + firm_price_factor)
            elif self._price_change == "avg_price":
                firm_price = math.exp(math.log(self.model.price + 1e-8) + firm_price_factor)
            else:
                raise ValueError(
                    f"Invalid price change mechanism. Must be 'agent_price' or 'avg_price'. Got {self._price_change} "
                    f"This should not happen, as it is checked in the __init__ method."
                )

            # adjust (minium demand is alpha = at least 1 worker)
            if exp_demand < self.model.params["alpha"]:
                exp_demand = self.model.params["alpha"]

            # modify in the julia model
            self.model[f_id].De = exp_demand
            self.model[f_id].P = firm_price
