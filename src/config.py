"""
Environment Configurations
"""
import numpy as np
from data_simulation import generate_lead_time, generate_prod_capacity
from data_simulation import generate_cost_price, generate_sup_dem_relations
from data_simulation import generate_holding_costs, generate_backlog_costs, generate_init_inventories
from data_simulation import Demand_fn
import os
from utils import save_dict_to_json

np.random.seed(0)

env_configs = {
    "test": {
        "config_name": "test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 1,
        "num_init_customers": 1,
        "num_agents_per_stage": 4, # >= 2
        "num_periods": 16,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": "constant", # constant/uniform/etc
        "price_cost_dist": "constant", # constant/uniform/normal/etc
        "lead_time_dist": "constant", # constant/uniform
        "prod_capacity_dist": "constant", # constant/uniform
        "demand_fn": "constant_demand", # constant/functional
        "holding_costs_dist": "constant",
        "backlog_costs_dist": "constant",
        "profit_rate_dist": "constant",
    },
    "basic": {
        "config_name": "basic",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 1,
        "num_init_customers": 1,
        "num_agents_per_stage": 4, # >= 2
        "num_periods": 2, 
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": ("uniform", 10, 15), # constant/uniform/etc
        "price_cost_dist": "uniform", # constant/uniform/normal/etc
        "lead_time_dist": ("uniform", 1, 5), # constant/uniform
        "prod_capacity_dist": ("uniform", 10, 20), # constant/uniform
        "demand_fn": ("constant_demand", 5), # constant/functional
        "holding_costs_dist": "constant", 
        "backlog_costs_dist": "constant", 
        "profit_rate_dist": ("uniform", 0, 1), 
    }, 
    "large_graph_test": {
        "config_name": "large_graph_test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 1,
        "num_init_customers": 1,
        "num_agents_per_stage": 20, # >= 2
        "num_periods": 8, 
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": ("uniform", 10, 15), # constant/uniform/etc
        "price_cost_dist": "uniform", # constant/uniform/normal/etc
        "lead_time_dist": ("uniform", 1, 5), # constant/uniform
        "prod_capacity_dist": ("uniform", 10, 20), # constant/uniform
        "demand_fn": ("constant_demand", 5), # constant/functional
        "holding_costs_dist": "constant", 
        "backlog_costs_dist": "constant", 
        "profit_rate_dist": ("uniform", 0, 1), 
    }
}

def get_env_configs(env_configs: dict):
    
    save_dict_to_json(data=env_configs, save_path=env_configs['config_name'])
    num_stages = env_configs["num_stages"]
    num_agents_per_stage = env_configs["num_agents_per_stage"]
    num_periods = env_configs["num_periods"]
    num_total_agents = num_stages * num_agents_per_stage
    
    supply_relations, demand_relations = \
        generate_sup_dem_relations(type=env_configs["sup_dem_relation_type"], num_stages=num_stages, num_agents_per_stage=num_agents_per_stage, \
                                   num_suppliers=env_configs["num_init_suppliers"], num_customers=env_configs["num_init_customers"])
    order_costs, sale_prices = \
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


    return {
        'num_stages': num_stages,
        'num_periods': num_periods,
        'num_agents_per_stage': num_agents_per_stage,
        "demand_dist": env_configs["demand_fn"][0],
        'init_inventories': init_inventories, # num_stages * num_agents_per_stage
        'lead_times': lead_times, # num_stages * num_agents_per_stage * num_agents_per_stage
        'demand_fn': demand_fn,
        'prod_capacities': prod_capacities,
        'sale_prices': sale_prices,
        'order_costs': order_costs,
        'backlog_costs': backlog_costs,
        'holding_costs': holding_costs,
        'supply_relations': supply_relations,
        "demand_relations": demand_relations,
        'stage_names': stage_names,
    }
    

