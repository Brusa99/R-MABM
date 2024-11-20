#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example of a heuristic run of the Cats environment with a single dummy agent.
Last updated: 2024-11-20
Version: 0.0.4
"""

import matplotlib.pyplot as plt
import numpy as np

from rmabm.environments import Cats
from rmabm.agents import Dummy
from rmabm import Simulation, Logger


# Create a single agent environment
env = Cats(n_agents=1)

# Create an agent to gather baseline data
agent = Dummy(env.agents_ids[0])

# You can run the simulation steps manually
obs_list, info = env.reset()
for i in range(100):
    action = agent.get_action(obs_list[0])  # note that the agent is a dummy so the observation is ignored
    obs_list, reward_list, terminated, truncated, info, = env.step([action])  # notice that action is a put in a list

    # Print information
    print(f"Step [{i}]: agent's profit = {reward_list[0]}, GDP = {info["Y_real"]}")

    if truncated or terminated:
        break

# Or you can use the Simulation class
sim = Simulation(environment=env, agents=[agent])

# By advancing by a step, the observations, rewards, terminated, truncated flags, and infos are saved internally.
for t in range(100):
    sim.step()

# We can access the history of observations, actions, rewards, and infos
print(sim.obs_history)
print(sim.act_history)  # this is not interesting for the dummy agent
print(sim.reward_history)
print(sim.info_history)

# We can use matplotlib to plot the data
plt.plot(sim.reward_history)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.show()

# For other histories, you have to preprocess the data, for example, to plot the GDP
gdp = [info["Y_real"] for info in sim.info_history]
plt.plot(gdp)
plt.xlabel("Step")
plt.ylabel("GDP")
plt.show()

# For multi-dimensional histories, like the observations, you can plot the data for each agent
first_agent_obs = [obs[0] for obs in sim.obs_history]
first_agent_stock = [obs["firm_stock"] for obs in first_agent_obs]
plt.plot(first_agent_stock)
plt.xlabel("Step")
plt.ylabel("Stock")
plt.show()

# We can use the Logger class to save the data to disk
logger = Logger(log_name="heruistic_run", log_directory="logs", use_timestamp=False)
logger.log_obs(sim.obs_history)
logger.log_act(sim.act_history)  # this will not log anything if only dummy agents are present
logger.log_array(np.array(sim.reward_history), "rewards")
logger.log_info(sim.info_history)

# The logger can be initialized directly in the Simulation class
sim.init_logger(log_name="sim_heruistic_run", log_directory="logs", use_timestamp=False)

# We can reset the simulation, this will delete histories and reset the environment
sim.reset()

# We can run the simulation for an entire episode
# If a logger is initialized, the data is logged with the current episode number as a file prefix
sim.run_episode()
