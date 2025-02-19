import os
import re
import sys
import time
import numpy as np
from typing import List
from tqdm.notebook import tqdm
sys.path.append('src')
# sys.path.append('gnn')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.config import env_configs, get_env_configs
from model.utils import clear_dir, split_demand
import json
from tqdm import tqdm
# np.random.seed(42)




save_datapath = "src/gnn/gnn_dataset"
env_config_name = "large_graph_test"

# create the dir to store the results
os.makedirs(f"results/{env_config_name}", exist_ok=True)
clear_dir(f"results/{env_config_name}")
# crate the dir to store the env setup
os.makedirs(f"env/{env_config_name}", exist_ok=True)
clear_dir(f"env/{env_config_name}")
env_config = get_env_configs(env_configs=env_configs[env_config_name])
demand_fn = env_config["demand_fn"]


num_period = 50
num_stages = env_config["num_stages"]
num_agents_per_stage = env_config["num_agents_per_stage"]
num_init_suppliers = env_config['num_init_suppliers']
num_agents = num_stages * num_agents_per_stage

init_inventories = env_config["init_inventories"]
lead_times = env_config["lead_times"]
prod_capacities = env_config["prod_capacities"]
sale_prices = env_config["sale_prices"]
order_costs = env_config["order_costs"]
prod_costs = env_config["prod_costs"]
holding_costs = env_config["holding_costs"]
backlog_costs = env_config["backlog_costs"]

dataset = []
for i in tqdm(range(600)):

    stage_id = i%(num_stages-2)
    demand_seq = []
    for t in np.arange(num_period):
        demand = int(demand_fn(t)) + np.random.normal(0, demand_fn.mean//2)
        demand = max(demand, 0)
        demands = [int(x) for x in split_demand(demand=demand, num_suppliers=num_init_suppliers, num_agents_per_stage=num_agents_per_stage)]
        demand_seq.append(demands)

    fulfilled_rate = np.round(np.random.normal(0.8, 0.3, size=num_agents_per_stage), 3)
    fulfilled_rate[fulfilled_rate>1] = 1
    fulfilled_rate[fulfilled_rate<0] = 0
    data = {
            "stage_id": stage_id + 1, 
            "demands": demand_seq, 
            "inventory": init_inventories[i%num_agents].tolist(), 
            "lead_times": lead_times[stage_id+1, i%num_agents_per_stage, :].tolist(), 
            "prod_capacity": prod_capacities[i%num_agents].tolist(),
            "sale_price": sale_prices[(stage_id+1)*num_agents_per_stage+i%num_agents_per_stage].tolist(),
            "order_costs": order_costs[(stage_id+1)*num_agents_per_stage: (stage_id+2)*num_agents_per_stage].tolist(), 
            "prod_cost": prod_costs[i%num_agents].tolist(), 
            "holding_cost": holding_costs[i%num_agents].tolist(),
            "backlog_cost": backlog_costs[i%num_agents].tolist(),
            "fulfilled_rate": fulfilled_rate.tolist(), 
            }
    dataset.append(data)

os.makedirs(f"{save_datapath}", exist_ok=True)
print(f"Writing data to {save_datapath}/gnn_data_Env({env_config_name}).json")
with open(f"{save_datapath}/gnn_data_Env({env_config_name}).json", 'w') as f:
    json.dump(dataset, f, indent=4)








