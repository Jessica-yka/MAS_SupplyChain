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
from mas_model import create_agents
from mas_model import run_simulation
from utils import get_demand_description, get_state_description
from utils import clear_dir
np.random.seed(42)
# TO-DO



env_config_name = "large_graph_test"
# create the dir to store the results
os.makedirs(f"results/{env_config_name}", exist_ok=True)
clear_dir(f"results/{env_config_name}")
# create the dir to store the env setup
os.makedirs(f"env/{env_config_name}", exist_ok=True)
clear_dir(f"env/{env_config_name}")
env_config = get_env_configs(env_configs=env_configs[env_config_name])
demand_fn = env_config["demand_fn"]

period = np.random.choice(50, 500)
demand = 0
