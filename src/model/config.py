"""
Environment Configurations
"""
import numpy as np
import sys
sys.path.append('/data/yanjia/MAS_SupplyChain')
from src.model.data_simulation import generate_lead_time, generate_prod_capacity
from src.model.data_simulation import generate_cost_price, generate_sup_dem_relations
from src.model.data_simulation import generate_holding_costs, generate_backlog_costs, generate_init_inventories
from src.model.data_simulation import Demand_fn
import os
from src.model.utils.utils import save_dict_to_json, clear_dir
from collections import defaultdict

np.random.seed(0)

env_configs_list = {
    "large_graph_test": {
        "config_name": "large_graph_test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 2,
        "num_init_customers": 2,
        "num_agents_per_stage": 10, # >= 2
        "num_periods": 15,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": {'dist': "uniform", 'lb': 10, 'ub': 15}, # constant/uniform/etc
        "price_cost_dist": {'dist': 'uniform', 'lb': 1, 'ub': 8}, # constant/uniform/normal/etc
        "lead_time_dist": {'dist': 'uniform', 'lb': 2, 'ub': 15}, # constant/uniform
        "prod_capacity_dist": {'dist': 'uniform', 'lb': 25, 'ub': 40}, # constant/uniform("uniform", 25, 40)
        "demand_fn": {"dist": "constant_demand", "mean": 5, "trend": "linear", 'with_noise': True}, # constant/functional
        "holding_costs_dist": {"dist": "constant", "mean": 10}, 
        "backlog_costs_dist": {'dist': "constant", "mean": 5}, 
        "profit_rate_dist": {"dist": "uniform", "lb": 0, "ub": 1}, 
        "llm_agents": [(1, 1)],
        "enable_graph_change": True, 
        "enable_price_change": False, 
        "state_format": "base", 
        "env_no_backlog": True, 
        "emergent_events": {0: {"events": ["earthquake"], 'affected_agents': [[(2, 10)]]},
                            }, 

    },
    "large_graph_test_ee": {
        "config_name": "large_graph_test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 2,
        "num_init_customers": 2,
        "num_agents_per_stage": 20, # >= 2
        "num_periods": 10,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": ("uniform", 10, 15), # constant/uniform/etc
        "price_cost_dist": "uniform", # constant/uniform/normal/etc
        "lead_time_dist": ("uniform", 1, 10), # constant/uniform
        "prod_capacity_dist": ("uniform", 10, 80), # constant/uniform
        "demand_fn": {"dist": "constant_demand", "mean": 10, "trend": ""}, # constant/functional
        "holding_costs_dist": "constant", 
        "backlog_costs_dist": "constant", 
        "profit_rate_dist": ("uniform", 0, 1), 
        "llm_agents": [(1, 1)],
        "enable_graph_change": True, 
        "enable_price_change": False, 
        "state_format": "base", 
        "emergent_events": {}, 

    },
}

def get_env_configs(env_configs: dict):

    env_config_name = env_configs['config_name']
    # create the dir to store the results

    os.makedirs(f"results/{env_config_name}", exist_ok=True)
    clear_dir(f"results/{env_config_name}")
    # crate the dir to store the env setup
    os.makedirs(f"env/{env_config_name}", exist_ok=True)
    clear_dir(f"env/{env_config_name}")
    

    save_dict_to_json(data=env_configs, save_path=f"env/{env_config_name}/config.json")
    num_stages = env_configs["num_stages"]
    num_agents_per_stage = env_configs["num_agents_per_stage"]
    num_periods = env_configs["num_periods"]
    num_total_agents = num_stages * num_agents_per_stage
    num_init_suppliers = env_configs["num_init_suppliers"]


    supply_relations, demand_relations = \
        generate_sup_dem_relations(type=env_configs["sup_dem_relation_type"], num_stages=num_stages, num_agents_per_stage=num_agents_per_stage, \
                                   num_suppliers=env_configs["num_init_suppliers"], num_customers=env_configs["num_init_customers"])
    order_costs, sale_prices, prod_costs = \
        generate_cost_price(prod_cost_dist=env_configs["price_cost_dist"], profit_rate_dist=env_configs["profit_rate_dist"], \
                            num_stages=num_stages, num_agents_per_stage=num_agents_per_stage, config_name=env_configs["config_name"])
    holding_costs = \
        generate_holding_costs(dist=env_configs["holding_costs_dist"], num_data=num_total_agents, config_name=env_configs["config_name"])
    backlog_costs = \
        generate_backlog_costs(dist=env_configs["backlog_costs_dist"], num_data=num_total_agents, config_name=env_configs["config_name"])
    lead_times = \
        generate_lead_time(dist=env_configs["lead_time_dist"], num_stages=num_stages, num_agents_per_stage=num_agents_per_stage,config_name=env_configs["config_name"])
    prod_capacities = \
        generate_prod_capacity(dist=env_configs['prod_capacity_dist'], num_data=num_total_agents, config_name=env_configs["config_name"])
    init_inventories = \
        generate_init_inventories(dist=env_configs["init_inventory_dist"], num_data=num_total_agents, config_name=env_configs["config_name"])
    # profit_rates = \
    #     generate_profit_rates(dist=env_configs["profit_rate_dist"], num_data=num_total_agents, config_name=env_configs["config_name"])
    
    demand_fn = Demand_fn(dist=env_configs["demand_fn"])
    stage_names = env_configs["stage_names"]
    llm_agents = env_configs["llm_agents"]
    state_format = env_configs["state_format"]
    env_no_backlog = env_configs["env_no_backlog"]
    
    enable_graph_change = env_configs["enable_graph_change"]
    enable_price_change = env_configs["enable_price_change"]
    emergent_events = env_configs["emergent_events"]


    return {
        'num_stages': num_stages,
        'num_periods': num_periods,
        'num_agents_per_stage': num_agents_per_stage,
        "demand_dist": env_configs["demand_fn"]["dist"],
        'init_inventories': init_inventories, # num_stages * num_agents_per_stage
        'lead_times': lead_times, # num_stages * num_agents_per_stage * num_agents_per_stage
        'demand_fn': demand_fn,
        'prod_capacities': prod_capacities,
        'sale_prices': sale_prices,
        'order_costs': order_costs,
        "prod_costs": prod_costs, 
        'backlog_costs': backlog_costs,
        'holding_costs': holding_costs,
        'num_init_suppliers': num_init_suppliers, 
        'supply_relations': supply_relations,
        "demand_relations": demand_relations,
        'stage_names': stage_names,
        "llm_agents": llm_agents,
        "state_format": state_format, 
        "env_no_backlog": env_no_backlog, 
        "enable_graph_change": enable_graph_change,
        "enable_price_change": enable_price_change, 
        "emergent_events": emergent_events,
    }
    

