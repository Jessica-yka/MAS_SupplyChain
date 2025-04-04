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
from src.model.gpt_mas_model import create_agents as create_gpt_agents
from src.model.gpt_mas_model import run_period_simulation as run_gpt_period_simulation
from src.model.llama_mas_model import create_agents as create_llama_agents
from src.model.llama_mas_model import run_period_simulation as run_llama_period_simulation, chat_with_llama_agents
from src.model.utils.utils import get_demand_description, get_state_description, clear_dir, visualize_state, create_action_dicts
from src.model.env import load_env_attributes, save_env_attributes, reverse_env_to_t
import argparse
import json

parser = argparse.ArgumentParser(description='Supply Chain Management')
parser.add_argument('--env_file', type=str, default=None, help='Environment Configuration')
parser.add_argument('--cur_period', type=int, default=None, help='Current Period')
# parser.add_argument('--llm_agent_model', type=str, default="gpt", help='LLM Agent Models (GPT series or LLAMA series)')

np.random.seed(42)

# ENV_CONFIG_NAME = "inference_only"
ENV_CONFIG_NAME = "graph_4_4"
LLM_AGENT_NAME = "llama"

# %% [markdown]
# ## Creating the Environment

# %%req orders 
def create_supply_chain_environment(env_json=None, cur_period: int=0):
    env_config = get_env_configs(env_configs=env_configs_list[ENV_CONFIG_NAME])
    im_env = env_creator(env_config)
    im_env.reset()

    if env_json is not None: # the simulation has started
        im_env = load_env_attributes(im_env, file_path=f"results/{ENV_CONFIG_NAME}/json_results/env.json")

        # print(t)
        if im_env.period > cur_period: # reverse timeline
            print("reverse the timeline to", cur_period)
            reverse_env_to_t(im_env, cur_period, ENV_CONFIG_NAME)

    return im_env, env_config


def period_simulation_framework(env_json=None, cur_period: int=0, events=[]):
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
        if os.path.exists("testing_llama_mas_question_formulation.csv"):
            os.remove("testing_llama_mas_question_formulation.csv")
        # clear_dir(f"env/{ENV_CONFIG_NAME}")
        if os.path.exists(f"llama_mas_user_chat_history.csv"):
            os.remove(f"llama_mas_user_chat_history.csv")

    # setup the environment
    im_env, env_config = create_supply_chain_environment(env_json=env_json, cur_period=cur_period)
    # create the agents
    stage_agents = None
    if LLM_AGENT_NAME == "gpt":
        stage_agents, user_proxy = create_gpt_agents(env_config["stage_names"], env_config["max_num_agents_per_stage"], llm_config={"config_list": llm_config_list})
    elif LLM_AGENT_NAME == "llama":
        stage_agents = create_llama_agents(env_config["num_stages"], env_config["max_num_agents_per_stage"])
        pass
    
    if env_json is not None: # the env has been running.      
        print("calculate the env update, rewards, etc")
        im_env = update_user_change_order_from_json(im_env, env_json=env_json)
        next_states, rewards, terminations, truncations, infos = im_env.step()
        im_env = update_user_change_attribute_from_json(im_env, env_json=env_json, events=events)

    # run the simulation
    print("run simulation")
    if LLM_AGENT_NAME == "gpt":
        im_env, env_json = run_gpt_period_simulation(im_env=im_env, user_proxy=user_proxy, stage_agents=stage_agents, config_name=ENV_CONFIG_NAME)
    elif LLM_AGENT_NAME == "llama":
        im_env, env_json = run_llama_period_simulation(im_env=im_env, stage_agents=stage_agents, events=events, config_name=ENV_CONFIG_NAME)
    # save the env attributes
    save_env_attributes(im_env, file_path=f"results/{ENV_CONFIG_NAME}/json_results/env.json")

    return env_json

def user_conversation_with_llama_agents(query: list, cur_period: int=0):
    # read env from json
    env_json_path = f"results/{ENV_CONFIG_NAME}/json_results/env_period_{cur_period}.json"
    if os.path.exists(env_json_path):
        with open(env_json_path, 'r') as f:
            env_json = json.load(f)
    else:
        raise FileNotFoundError(f"Environment JSON file not found: {env_json_path}")
    # setup the environment
    im_env, env_config = create_supply_chain_environment(env_json=env_json, cur_period=cur_period)
    # create the agents
    stage_agents = create_llama_agents(env_config["num_stages"], env_config["max_num_agents_per_stage"])
    ans = chat_with_llama_agents(im_env=im_env, stage_agents=stage_agents, query=query)

    return ans


if __name__ == "__main__":

    args = parser.parse_args()
    # env_period_0.json
    if os.path.exists(f"results/{ENV_CONFIG_NAME}/json_results/{args.env_file}"):
        with open(f"results/{ENV_CONFIG_NAME}/json_results/{args.env_file}", 'r') as f:
            env_json = json.load(f)
    else:
        env_json = None

    cur_period = args.cur_period
    events = []
    # events = [{'id': 'event-1742998185720', 'type': 'Advances in operation robotics', 'effect': 'Positive', 'nodeStage': 0, 'nodeIndex': 0, 'companyType': 'Retailer'}]
    # query = [{"stage": 0, "agent_idx": 0, "message": "Tell me your reasoning process when placing the order."},]
    period_simulation_framework(env_json=env_json, cur_period=cur_period, events=events)
    # ans = user_conversation_with_llama_agents(cur_period=cur_period, query=query)
    # print(ans)