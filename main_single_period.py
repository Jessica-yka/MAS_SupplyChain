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
# sys.path.append('src')
from src.model.env import env_creator, update_user_change_attribute_from_json, update_user_change_order_from_json
from src.model.config import env_configs_list, get_env_configs
from src.model.llm_config import llm_config_list
from openai import AzureOpenAI
from src.model.gpt_mas_model import create_agents
from src.model.gpt_mas_model import run_period_simulation
from src.model.utils.utils import get_demand_description, get_state_description, clear_dir, visualize_state, create_action_dicts
from src.model.utils.utils import add_xy_locations_to_env_json
from src.model.env import load_env_attributes, save_env_attributes, reverse_env_to_t
import argparse
import json

parser = argparse.ArgumentParser(description='Supply Chain Management')
parser.add_argument('--env_file', type=str, default=None, help='Environment Configuration')
parser.add_argument('--cur_period', type=int, default=None, help='Current Period')

np.random.seed(42)

ENV_CONFIG_NAME = "graph_4_4"

# %% [markdown]
# ## Creating the Environment

# %%req orders 
def create_supply_chain_environment(env_json=None, cur_period: int=0):
    env_config = get_env_configs(env_configs=env_configs_list[ENV_CONFIG_NAME])
    im_env = env_creator(env_config)
    im_env.reset()

    action_dicts = {}
    if env_json is not None: # the simulation has started
        im_env = load_env_attributes(im_env, file_path=f"results/{ENV_CONFIG_NAME}/json_results/env.json")

        # print(t)
        if im_env.period > cur_period: # reverse timeline
            print("reverse the timeline to", cur_period)
            reverse_env_to_t(im_env, cur_period, ENV_CONFIG_NAME)

    # else: # the simulation is just setup
        # create the dir to store the env setup and results
        # create_action_dicts(ENV_CONFIG_NAME)

    # action_dicts = {"action_order_dict": action_order_dict, "action_sup_dict": action_sup_dict, "action_dem_dict": action_dem_dict, "action_price_dict": action_price_dict,
    #                 "rewards": {}, "states": {}}

    return im_env, env_config


def period_simulation_framework(env_json=None, cur_period: int=0):
    # prepare the simulation with direction creation
    if env_json is None:
        os.makedirs(f"results/{ENV_CONFIG_NAME}", exist_ok=True)
        os.makedirs(f"results/{ENV_CONFIG_NAME}/json_results", exist_ok=True)
        os.makedirs(f"results/{ENV_CONFIG_NAME}/img_results", exist_ok=True)
        os.makedirs(f"results/{ENV_CONFIG_NAME}/df_results", exist_ok=True)
        os.makedirs(f"results/{ENV_CONFIG_NAME}/chat_results", exist_ok=True)
        os.makedirs(f"env/{ENV_CONFIG_NAME}", exist_ok=True)
        
        clear_dir(f"results/{ENV_CONFIG_NAME}/json_results")
        clear_dir(f"results/{ENV_CONFIG_NAME}/img_results")
        clear_dir(f"results/{ENV_CONFIG_NAME}/df_results")
        clear_dir(f"results/{ENV_CONFIG_NAME}/chat_results")
        clear_dir(f"env/{ENV_CONFIG_NAME}")

    # setup the environment
    im_env, env_config = create_supply_chain_environment(env_json=env_json, cur_period=cur_period)
    # create the agents
    stage_agents, user_proxy = create_agents(env_config["stage_names"], env_config["num_agents_per_stage"], llm_config={"config_list": llm_config_list})
    
    if env_json is not None: # the env has been running.      
        print("calculate the env update, rewards, etc")
        im_env = update_user_change_order_from_json(im_env, env_json=env_json)
        next_states, rewards, terminations, truncations, infos = im_env.step()
        im_env = update_user_change_attribute_from_json(im_env, env_json=env_json)
    # run the simulation
    print("run simulation")
    im_env, env_json = run_period_simulation(im_env=im_env, user_proxy=user_proxy, stage_agents=stage_agents, config_name=ENV_CONFIG_NAME)
    # save the env attributes
    save_env_attributes(im_env, file_path=f"results/{ENV_CONFIG_NAME}/json_results/env.json")

    return env_json



if __name__ == "__main__":

    args = parser.parse_args()
    # env_period_0.json
    with open(f"results/{ENV_CONFIG_NAME}/json_results/{args.env_file}", 'r') as f:
        env_json = json.load(f)
    cur_period = args.cur_period
    period_simulation_framework(env_json=env_json, cur_period=cur_period)