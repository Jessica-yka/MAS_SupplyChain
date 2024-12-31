"""
Environment Configurations
"""
import numpy as np
from utils import random_relations

np.random.seed(0)

num_stages = 4
num_agents_per_stage = 3
num_total_agents = num_stages * num_agents_per_stage
num_periods = 1
stage_names = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
supply_relations = {} # who are my suppliers
demand_relations = {} # who are my customers
for m in range(num_stages):
    supply_relations[m] = dict()
    demand_relations[m] = dict()
    for x in range(num_agents_per_stage):
        if m == 0: 
            supply_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int) 
            supply_relations[m][x][x] = 1
            demand_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int) # retailers have no downstream company
        elif m == num_stages-1: 
            supply_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int) # manufacturers have no upstream company
            demand_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
            demand_relations[m][x][x] = 1
        else:
            supply_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
            supply_relations[m][x][x] = 1
            demand_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
            demand_relations[m][x][x] = 1



env_configs = {
    'constant_demand': {
        'num_stages': num_stages,
        'num_periods': num_periods,
        'num_agents_per_stage': num_agents_per_stage,
        'init_inventories': [12 for _ in range(num_total_agents)], # num_stages * num_agents_per_stage
        'lead_times': [2 for _ in range(num_total_agents)],
        'demand_fn': lambda t: 4,
        'prod_capacities': [20 for _ in range(num_total_agents)],
        'sale_prices': [0 for _ in range(num_total_agents)],
        'order_costs': [0 for _ in range(num_total_agents)],
        'backlog_costs': [1 for _ in range(num_total_agents)],
        'holding_costs': [1 for _ in range(num_total_agents)],
        'supply_relations': supply_relations,
        "demand_relations": demand_relations,
        'stage_names': stage_names,
    },
    'two_agent': {
        'num_stages': 2,
        'num_agents_per_stage': 2, 
        'num_periods': 2,
        'init_inventories': [4 for _ in range(4)],
        'lead_times': [1, 2, 3, 4],
        'demand_fn': lambda t: 4,
        'prod_capacities': [10 for _ in range(4)],
        'sale_prices': [0 for _ in range(4)],
        'order_costs': [0 for _ in range(4)],
        'backlog_costs': [1 for _ in range(4)],
        'holding_costs': [1 for _ in range(4)],
        'stage_names': ['retailer', 'supplier'],
    },
    'variable_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: np.random.randint(0, 5),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [0, 0, 0, 0],
        'order_costs': [0, 0, 0, 0],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
    },
    'larger_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: np.random.randint(0, 9),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
    },
    'seasonal_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: np.random.randint(0, 5) if t <= 4 else np.random.randint(5, 9),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
    },
    'normal_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 14, 16, 18],
        'lead_times': [1, 2, 3, 4],
        'demand_fn': lambda t: max(0, int(np.random.normal(4, 2))),
        'prod_capacities': [20, 22, 24, 26],
        'sale_prices': [9, 8, 7, 6],
        'order_costs': [8, 7, 6, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
    },
}

