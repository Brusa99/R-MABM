#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from rmabm.environments import Cats
from rmabm.agents import Dummy
from rmabm import Simulation

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
sim.run_episode()
