# %% [markdown]
# # AutoGen for Supply Chain Management

# %%
import os
import re
import sys
import time
import numpy as np
from typing import List
from tqdm import tqdm
from autogen import ConversableAgent
sys.path.append('src')
from env import env_creator
from config import env_configs, get_env_configs
from llm_config import llm_config_list
from openai import AzureOpenAI
from mas_model import create_agents
from mas_model import run_simulation
from utils import get_demand_description, get_state_description
from utils import clear_dir
np.random.seed(42)

# %%


# %%
config_list = llm_config_list


# %% [markdown]
# ## Creating the Environment

# %%req orders 
env_config_name = "large_graph_test"
# create the dir to store the results
os.makedirs(f"results/{env_config_name}", exist_ok=True)
clear_dir(f"results/{env_config_name}")
# create the dir to store the env setup
os.makedirs(f"env/{env_config_name}", exist_ok=True)
clear_dir(f"env/{env_config_name}")
env_config = get_env_configs(env_configs=env_configs[env_config_name])
im_env = env_creator(env_config)

# %% [markdown]
# ## Getting Descriptions

# %%
print(env_config["demand_dist"])
print(get_demand_description(env_config["demand_fn"]))

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
for r in tqdm(range(1)):
    print("\n\nNew round starts")
    stage_agents = create_agents(stage_names=env_config["stage_names"], num_agents_per_stage=env_config['num_agents_per_stage'], llm_config={'config_list':config_list})
    reward = run_simulation(im_env=im_env, user_proxy=user_proxy, stage_agents=stage_agents, config_name=env_config_name, round=r)
    rewards.append(reward)
    print(f"rewards = {reward}")
    # if reward < 0:
    #     raise AssertionError("The rewards are negative")

mean_reward = np.mean(rewards)
std_reward = np.std(rewards)

print(f"Rewards: {rewards}")
print(f"Mean Episode Reward: {mean_reward}")
print(f"Standard Deviation of Episode Reward: {std_reward}")

