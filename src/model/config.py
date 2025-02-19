"""
Environment Configurations
"""
import numpy as np
from model.data_simulation import generate_lead_time, generate_prod_capacity
from model.data_simulation import generate_cost_price, generate_sup_dem_relations
from model.data_simulation import generate_holding_costs, generate_backlog_costs, generate_init_inventories
from model.data_simulation import Demand_fn
import os
from model.utils import save_dict_to_json
from collections import defaultdict

np.random.seed(0)

env_configs = {
    "large_graph_test": {
        "config_name": "large_graph_test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 2,
        "num_init_customers": 2,
        "num_agents_per_stage": 30, # >= 2
        "num_periods": 20,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": ("uniform", 10, 15), # constant/uniform/etc
        "price_cost_dist": "uniform", # constant/uniform/normal/etc
        "lead_time_dist": ("uniform", 2, 8), # constant/uniform
        "prod_capacity_dist": ("uniform", 25, 40), # constant/uniform
        "demand_fn": {"dist": "constant_demand", "mean": 5, "trend": "linear"}, # constant/functional
        "holding_costs_dist": {"dist": "constant", "mean": 10}, 
        "backlog_costs_dist": "constant", 
        "profit_rate_dist": ("uniform", 0, 1), 
        "llm_agents": [(1, 1)],
        "enable_graph_change": True, 
        "enable_price_change": False, 
        "state_format": "base", 
        "env_no_backlog": True, 
        "emergent_events": [], 
        "shut_seq": {},
        "rec_seq": {},
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
        "emergent_events": [(5, "sudden_shutdown"), (7, "recovery")], 
        "shut_seq": {5: [(2, 2), (2, 10), (2, 13)]},
        "rec_seq": {7: [(2,2), (2,10)]},
    },
    "large_graph_normal_demand_test": {
        "config_name": "large_graph_test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 3,
        "num_init_customers": 3,
        "num_agents_per_stage": 20, # >= 2
        "num_periods": 8,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": ("uniform", 10, 15), # constant/uniform/etc
        "price_cost_dist": "uniform", # constant/uniform/normal/etc
        "lead_time_dist": ("uniform", 1, 10), # constant/uniform
        "prod_capacity_dist": ("uniform", 10, 80), # constant/uniform
        "demand_fn": {"dist": "normal_demand", "mean": 10, "std": 3, "trend": False}, # constant/functional
        "holding_costs_dist": "constant", 
        "backlog_costs_dist": "constant", 
        "profit_rate_dist": ("uniform", 0, 1), 
        "llm_agents": [(0, 1)],
        "enable_graph_change": True, 
        "state_format": "base", 
    },
}

def get_env_configs(env_configs: dict):
    
    save_dict_to_json(data=env_configs, save_path=env_configs['config_name'])
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
    
    demand_fn = Demand_fn(dist=env_configs["demand_fn"]['dist'], mean=env_configs["demand_fn"].get("mean", 0), std=env_configs["demand_fn"].get("std", 0), 
                          lb=env_configs["demand_fn"].get("lb", 0), ub=env_configs["demand_fn"].get("ub", 0), trend=env_configs["demand_fn"].get("trend", False))
    stage_names = env_configs["stage_names"]
    llm_agents = env_configs["llm_agents"]
    state_format = env_configs["state_format"]
    env_no_backlog = env_configs["env_no_backlog"]
    
    enable_graph_change = env_configs["enable_graph_change"]
    enable_price_change = env_configs["enable_price_change"]
    emergent_events = defaultdict(list)
    for (t, ee) in env_configs["emergent_events"]:
        emergent_events[t].append(ee)
    shut_seq = env_configs["shut_seq"]
    rec_seq = env_configs["rec_seq"]

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
        "shut_seq": shut_seq,
        "rec_seq": rec_seq,  
    }
    

