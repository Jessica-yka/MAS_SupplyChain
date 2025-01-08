# %% [markdown]
# # AutoGen for Supply Chain Management

# %%
import os
import re
import sys
import time
import numpy as np
from typing import List
from tqdm.notebook import tqdm
from autogen import ConversableAgent
sys.path.append('src')
from env import env_creator
from config import env_configs, get_env_configs
from llm_config import llm_config_list
from openai import AzureOpenAI
from model import create_agents
from model import run_simulation
from utils import get_demand_description, get_state_description

np.random.seed(42)

# %%


# %%
config_list = llm_config_list


# %% [markdown]
# ## Creating the Environment

# %%
env_config_name = "basic"
env_config = get_env_configs(env_configs=env_configs[env_config_name])
im_env = env_creator(env_config)

# %% [markdown]
# ## Getting Descriptions

# %%
print(env_config["demand_dist"])
print(get_demand_description(env_config["demand_dist"]))

# %% [markdown]
# ## Creating Agents

# %%
user_proxy = ConversableAgent(
    name="UserProxy",
    llm_config=False,
    human_input_mode="NEVER",
)

# %%
stage_agents = create_agents(env_config["stage_names"], env_config["num_agents_per_stage"], llm_config={"config_list": config_list})

# %%
# for stage_agent in stage_agents:
#     print(stage_agent.system_message)


# %% [markdown]
# ## Running Simulations

# %%
rewards = []

for _ in tqdm(range(1)):
    stage_agents = create_agents(stage_names=env_config["stage_names"], num_agents_per_stage=env_config['num_agents_per_stage'], llm_config={'config_list':config_list})
    reward = run_simulation(im_env, user_proxy, stage_agents)
    rewards.append(reward)
    print(f"rewards = {rewards}")

# mean_reward = np.mean(rewards)
# std_reward = np.std(rewards)

# print(f"Rewards: {rewards}")
# print(f"Mean Episode Reward: {mean_reward}")
# print(f"Standard Deviation of Episode Reward: {std_reward}")
