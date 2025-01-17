import numpy as np
import os
import re
from utils import save_array, extract_pairs
from utils import random_relations

def generate_lead_time(dist: tuple, num_stages: int, num_agents_per_stage: int, config_name: str="test"):
    assert len(dist) == 3 if dist[0] == 'uniform' else 1, "Please provide the lower bound and upper bound for the uniform distribution."
    assert len(dist) == 2 if dist[0] == 'constant' else 1, "Please provide the mean value for the constant distribution."
    
    # To generate lead time for each agent
    if dist[0] == 'uniform':
        data = np.random.uniform(low=dist[1], high=dist[2], size=(num_stages, num_agents_per_stage, num_agents_per_stage)).astype(int)
    elif dist[0] == "constant":
        mean = dist[1]
        data = [mean for _ in range(num_stages * num_agents_per_stage * num_agents_per_stage)]
        data = np.array(data).reshape(num_stages, num_agents_per_stage, num_agents_per_stage).astype(int)
    else:
        raise AssertionError("Lead time function is not implemented.")
    save_array(data, f"env/{config_name}/lead_time.npy")
    return data

def generate_prod_capacity(dist: tuple, num_data: int, config_name: str="test"):
    # To generate production capacity for each agent
    assert len(dist) == 3 if dist[0] == 'uniform' else 1, "Please provide the lower bound and upper bound for the uniform distribution."
    assert len(dist) == 2 if dist[0] == 'constant' else 1, "Please provide the mean value for the constant distribution."
    
    if dist[0] == 'uniform':
        data = np.random.uniform(low=dist[1], high=dist[2], size=num_data).astype(int)
    elif dist[0] == 'constant':
        data = np.array([dist[1] for _ in range(num_data)]).astype(int)
    else:
        raise AssertionError("Prod capacity function is not implemented.")
    
    save_array(data, f"env/{config_name}/prod_capacity.npy")
    return data


def generate_profit_rates(dist: tuple, num_data: int, config_name: str="test"):
    # To generate profit rate for agents to decide price based on cost
    assert len(dist) == 3 if dist[0] == 'uniform' else 1, "Please provide the lower bound and upper bound for the uniform distribution."
    assert len(dist) == 2 if dist[0] == 'constant' else 1, "Please provide the mean value for the constant distribution."

    if dist[0] == "uniform":
        data = 1 + np.random.uniform(low=dist[1], high=dist[2], size=num_data)
    elif dist[0] == 'constant':
        mean = 1 + dist[1]
        data = np.array([mean for _ in range(num_data)])
    else:
        raise AssertionError("Profit rate function is not implemented.")

    save_array(data, f"env/{config_name}/profit_rate.npy")
    return data

def generate_prod_cost(dist: str, num_data: int, lb=5, ub=15, config_name: str="test"):

    if dist == "uniform":
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    elif dist == "constant":
        mean = (lb + ub)//2
        data = np.array([mean for _ in range(num_data)])
    else:
        raise AssertionError("Prod cost function is not implemented.")
    save_array(data, f"env/{config_name}/prod_cost.npy")
    return data

def generate_cost_price(prod_cost_dist: str, profit_rate_dist: tuple, num_stages: int, num_agents_per_stage: int, config_name: str="test"):

    # price = total cost * profit rate
    # cost = order cost + production cost
    num_total_agents = num_stages * num_agents_per_stage

    all_profit_rate = generate_profit_rates(dist=profit_rate_dist, num_data=num_total_agents, config_name=config_name)
    all_prod_costs = generate_prod_cost(dist=prod_cost_dist, num_data=num_total_agents, config_name=config_name)

    all_sale_prices = []
    all_order_costs = []

    manu_prices = all_prod_costs[:num_agents_per_stage] * all_profit_rate[:num_agents_per_stage]
    all_sale_prices += manu_prices.tolist() # add prices of manufacturers to the price list
    all_order_costs += [0 for _ in range(num_agents_per_stage)] # add cost of manufacturers to the cost list
    for i in range(1, num_stages):
        order_costs = all_sale_prices[:num_agents_per_stage]
        prod_costs = all_prod_costs[i*num_agents_per_stage:(i+1)*num_agents_per_stage]
        profit_rate = all_profit_rate[i*num_agents_per_stage:(i+1)*num_agents_per_stage]
        sale_prices = ((np.max(order_costs) + prod_costs) * profit_rate)
        all_sale_prices = sale_prices.tolist() + all_sale_prices
        all_order_costs = order_costs + all_order_costs

    all_sale_prices = np.array(all_sale_prices).astype(int)
    all_order_costs = np.array(all_order_costs).astype(int)
    save_array(all_sale_prices, f"env/{config_name}/sale_prices.npy")
    save_array(all_order_costs, f"env/{config_name}/total_costs.npy")
    return all_order_costs, all_sale_prices


def generate_sup_dem_relations(type: str, num_stages: int, num_agents_per_stage: int, \
                               num_suppliers: int=1, num_customers: int=1):
    if type == "single":
        supply_relations = {} # who are my suppliers
        demand_relations = {} # who are my customers
        for m in range(num_stages):
            supply_relations[m] = dict()
            demand_relations[m] = dict()
            for x in range(num_agents_per_stage):
                if m == 0: 
                    supply_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int) 
                    supply_relations[m][x][x] = 1
                    demand_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int) # assume that all retailers share the same downstream customer
                    demand_relations[m][x][0] = 1
                elif m == num_stages-1: 
                    supply_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int) # assume that all manufacturers share the same upstream supplier
                    supply_relations[m][x][0] = 1
                    demand_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
                    demand_relations[m][x][x] = 1
                else:
                    supply_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
                    supply_relations[m][x][x] = 1
                    demand_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
                    demand_relations[m][x][x] = 1
    elif type == "random":
        supply_relations = {}
        demand_relations = {}
        for m in range(num_stages):
            supply_relations[m] = dict()
            demand_relations[m] = dict()
            for x in range(num_agents_per_stage):
                if m == 0:
                    supply_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
                    suppliers_idx = random_relations(n_cand=num_agents_per_stage, n_relation=num_suppliers)
                    supply_relations[m][x][suppliers_idx] = 1
                    demand_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
                    demand_relations[m][x][0] = 1
                elif m == num_stages-1:
                    supply_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
                    supply_relations[m][x][0] = 1
                    demand_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
                    customers_idx = random_relations(n_cand=num_agents_per_stage, n_relation=num_customers)
                    demand_relations[m][x][customers_idx] = 1
                else:
                    supply_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
                    suppliers_idx = random_relations(n_cand=num_agents_per_stage, n_relation=num_suppliers)
                    supply_relations[m][x][suppliers_idx] = 1
                    demand_relations[m][x] = np.zeros(num_agents_per_stage, dtype=int)
                    customers_idx = random_relations(n_cand=num_agents_per_stage, n_relation=num_customers)
                    demand_relations[m][x][customers_idx] = 1

    else:
        raise AssertionError(f"{type} relation function is not implemented.")
    return supply_relations, demand_relations
    

def generate_holding_costs(dist: str, num_data: int, lb: int=1, ub: int=5, config_name: str="test"):

    if dist == 'constant':
        mean = (lb + ub)//2
        data = np.array([mean for _ in range(num_data)])
    elif dist == "uniform":
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    else:
        raise AssertionError("holding function is not implemented.")

    save_array(data, f"env/{config_name}/holding_costs.npy")
    return data


def generate_backlog_costs(dist: str, num_data: int, lb: int=1, ub: int=5, config_name: str="test"):

    if dist == 'constant':
        mean = (lb + ub)//2
        data = np.array([mean for _ in range(num_data)])
    elif dist == "uniform":
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    else:
        raise AssertionError("backlog function is not implemented.")
    
    save_array(data, f"env/{config_name}/backlog_costs.npy")
    return data
    

def generate_init_inventories(dist: tuple, num_data: int, config_name: str="test"):
    assert len(dist) == 3 if dist[0] == 'uniform' else 1, "Please provide the lower bound and upper bound for the uniform distribution."
    assert len(dist) == 2 if dist[0] == 'constant' else 1, "Please provide the mean value for the constant distribution."
    
    if dist[0] == "constant":
        mean = dist[1]
        data = np.array([mean for _ in range(num_data)]).astype(int)
    elif dist[0] == 'uniform':
        data = np.random.uniform(low=dist[1], high=dist[2], size=num_data).astype(int)
    else:
        raise AssertionError("init inventories is not implemented")
    
    save_array(data, f"env/{config_name}/init_inventories.npy")
    return data

class Demand_fn:

    def __init__(self, dist: tuple):
        assert len(dist) == 3 if dist[0] == 'uniform_demand' else 1, "Please provide the lower bound and upper bound for the uniform distribution."
        assert len(dist) == 2 if dist[0] == 'constant_demand' else 1, "Please provide the mean value for the constant distribution."
    
        self.dist = dist[0]
        if self.dist == 'uniform_demand':
            self.lb = dist[1]
            self.ub = dist[2]
            self.mean = None
        elif self.dist == 'constant_demand':
            self.mean = dist[1]
            self.lb = None
            self.ub = None

        self.period = -1

    def constant_demand(self):
        return self.mean

    def uniform_demand(self):
        return np.random.randint(low=self.lb, high=self.ub)
        
    def __call__(self, t):
        self.period = t
        if self.dist == 'constant_demand':
            return self.constant_demand()
        elif self.dist == "uniform_demand":
            return self.uniform_demand()
        else:
            raise AssertionError("Demand function is not implemented.")
        
