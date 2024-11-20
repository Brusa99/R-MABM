#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example of a training run of the CatsLog environment with two Q-Learning agents.
Last updated: 2024-11-20
Version: 0.0.5
"""

import os

import numpy as np

from rmabm.environments import CatsLog
from rmabm.agents import QLearner
from rmabm import Simulation

# Create an environment with two Q-Learning agents
env = CatsLog(n_agents=2)
agents = [QLearner(agent_id=agent_id, environment=env) for agent_id in env.agents_ids]

# The agents have a train method to update their Q-Matrix at each step
obs_list, info = env.reset()
for i in range(100):
    # get the actions of each agent
    actions = [agent.get_action(obs_list[index]) for index, agent in enumerate(agents)]
    # take a step in the environment
    next_obs_list, reward_list, terminated, truncated, info, = env.step(actions)
    # train the agents
    for index, agent in enumerate(agents):
        agent.train(obs_list[index], reward_list[index], next_obs_list[index], terminated)
    # update the observations
    obs_list = next_obs_list

# The Q-Matrices are now not all zeros
for agent in agents:
    print(f"agent {agent.agent_id} Q_max: {agent.Q.max()}, Q_min: {agent.Q.min()}")

# Note that the agents' learning rate (alpha) and exploration rate (epsilon) are not regulated by the agents
# This means that you should adjust these parameters, usually between episodes
env.reset()
for agent in agents:
    agent.epsilon *= 0.9  # decrease the exploration rate
    agent.alpha = agent.alpha  # keep constant learning rate

# You can use the Simulation class to simplify the training process
sim = Simulation(environment=env, agents=agents, n_episodes=5)

# the following line will train the agents for one episode
sim.train_agents(n_episodes=1)


# You can specify the update function to change the agents' parameters
def update(agent, simulation):
    agent.alpha = 0.2  # set the learning rate to 0.2

    # The simulation object is passed to the update function, so you can use it to update the agents' parameters
    agent.epsilon -= 0.1 * simulation.current_episode


# the update function will be called at the end of each episode
# if a number of episodes is not specified, it will train until the current episode number reaches the number of
#   episodes set at initialization
print("alpha before training update:", agents[0].alpha)  # 0.1
sim.train_agents(update=update)
print("alpha after training update:", agents[0].alpha)  # 0.2

# You can save the agents' Q-Matrices to a file using numpy
for agent in agents:
    os.makedirs("logs", exist_ok=True)
    np.save(f"logs/agent_{agent.agent_id}_Q.npy", agent.Q)
