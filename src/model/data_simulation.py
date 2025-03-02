import numpy as np
import os
import re
import sys
sys.path.append('/data/yanjia/MAS_SupplyChain')
from src.model.utils.utils import save_array, extract_pairs
from src.model.utils.utils import random_relations
from random import random

def generate_lead_time(dist: dict, num_stages: int, num_agents_per_stage: int, config_name: str="test", save_data: bool=True):
    # To generate lead time for each agent
    if dist['dist'] == 'uniform':
        data = np.random.uniform(low=dist['lb'], high=dist['ub'], size=(num_stages, num_agents_per_stage, num_agents_per_stage)).astype(int)
    elif dist['dist'] == "constant":
        mean = dist['mean']
        data = [mean for _ in range(num_stages * num_agents_per_stage * num_agents_per_stage)]
        data = np.array(data).reshape(num_stages, num_agents_per_stage, num_agents_per_stage).astype(int)
    else:
        raise AssertionError("Lead time function is not implemented.")

    if save_data:
        save_array(data, f"env/{config_name}/lead_time.npy")

    return data

def generate_prod_capacity(dist: dict, num_data: int, config_name: str="test", save_data: bool=True):
    # To generate production capacity for each agent
    if dist['dist'] == 'uniform':
        data = np.random.uniform(low=dist['lb'], high=dist['ub'], size=num_data).astype(int)
    elif dist['dist'] == 'constant':
        data = np.array([dist['mean'] for _ in range(num_data)]).astype(int)
    else:
        raise AssertionError("Prod capacity function is not implemented.")
    
    if save_data:
        save_array(data, f"env/{config_name}/prod_capacity.npy")
    return data


def generate_profit_rates(dist: dict, num_data: int, config_name: str="test", save_data: bool=True):
    # To generate profit rate for agents to decide price based on cost
    if dist['dist'] == "uniform":
        data = 1 + np.random.uniform(low=dist['lb'], high=dist['ub'], size=num_data)
    elif dist['dist'] == 'constant':
        mean = 1 + dist['mean']
        data = np.array([mean for _ in range(num_data)])
    else:
        raise AssertionError("Profit rate function is not implemented.")

    if save_data:
        save_array(data, f"env/{config_name}/profit_rate.npy")
    return data

def generate_prod_cost(dist: dict, num_data: int, lb=5, ub=15, config_name: str="test", save_data: bool=True):

    if dist['dist'] == "uniform":
        data = np.random.uniform(low=dist['lb'], high=dist['ub'], size=num_data)
    elif dist['dist'] == "constant":
        mean = dist['mean']
        data = np.array([mean for _ in range(num_data)])
    else:
        raise AssertionError("Prod cost function is not implemented.")
    data = data.astype(int)

    if save_data:
        save_array(data, f"env/{config_name}/prod_cost.npy")
    return data

def generate_cost_price(prod_cost_dist: dict, profit_rate_dist: dict, num_stages: int, num_agents_per_stage: int, 
                        config_name: str="test", save_data: bool=True):
    # price = total cost * profit rate
    # cost = order cost + production cost
    num_total_agents = num_stages * num_agents_per_stage

    all_profit_rate = generate_profit_rates(dist=profit_rate_dist, num_data=num_total_agents, config_name=config_name, save_data=save_data)
    all_prod_costs = generate_prod_cost(dist=prod_cost_dist, num_data=num_total_agents, config_name=config_name, save_data=save_data)

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

    if save_data:
        save_array(all_sale_prices, f"env/{config_name}/sale_prices.npy")
        save_array(all_order_costs, f"env/{config_name}/order_costs.npy")
    return all_order_costs, all_sale_prices, all_prod_costs


def generate_sup_dem_relations(type: str, num_stages: int, num_agents_per_stage: int, \
                               num_suppliers: int=1, num_customers: int=1):
    supply_relations = np.zeros((num_stages, num_agents_per_stage, num_agents_per_stage), dtype=int) # who are my suppliers
    demand_relations = np.zeros((num_stages, num_agents_per_stage, num_agents_per_stage), dtype=int) # who are my customers
    # Generate supply relations
    if type == "fix":
        for m in range(num_stages):
            for x in range(num_agents_per_stage):
                if m == 0: 
                    supply_relations[m][x][x] = 1
                    demand_relations[m][x][0] = 1
                elif m == num_stages-1: 
                    supply_relations[m][x][0] = 1
                else:
                    supply_relations[m][x][x] = 1
    elif type == "random":
        for m in range(num_stages):
            for x in range(num_agents_per_stage):
                if m == 0:
                    suppliers_idx = random_relations(n_cand=num_agents_per_stage, n_relation=num_suppliers)
                    supply_relations[m][x][suppliers_idx] = 1
                    demand_relations[m][x][0] = 1
                elif m == num_stages-1:
                    supply_relations[m][x][0] = 1
                else:
                    suppliers_idx = random_relations(n_cand=num_agents_per_stage, n_relation=num_suppliers)
                    supply_relations[m][x][suppliers_idx] = 1
    else:
        raise AssertionError(f"{type} relation function is not implemented.")
    
    # Infer demand relations from supply relations
    demand_relations[1:, :, :] = np.transpose(supply_relations[:-1, :, :], (0, 2, 1)) 
    
    return supply_relations, demand_relations
    

def generate_holding_costs(dist: dict, num_data: int, config_name: str="test", save_data: bool=True):

    if dist['dist'] == 'constant':
        mean = dist['mean']
        data = np.array([mean for _ in range(num_data)])
    elif dist == "uniform":
        lb = dist['lb']
        ub = dist['ub']
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    else:
        raise AssertionError("holding function is not implemented.")

    if save_data:
        save_array(data, f"env/{config_name}/holding_costs.npy")
    return data


def generate_backlog_costs(dist: dict, num_data: int, lb: int=1, ub: int=5, config_name: str="test", save_data: bool=True):
    if dist['dist'] == 'constant':
        mean = dist['mean']
        data = np.array([mean for _ in range(num_data)])
    elif dist['dist'] == "uniform":
        data = np.random.uniform(low=dist['lb'], high=dist['ub'], size=num_data)
    else:
        raise AssertionError("backlog function is not implemented.")
    
    if save_data:
        save_array(data, f"env/{config_name}/backlog_costs.npy")
    return data


def generate_backlogs(dist: dict, num_data: int, config_name: str="test", save_data: bool=False):
    
    if dist['dist'] == 'constant':
        mean = dist['mean']
        data = np.array([mean for _ in range(num_data)])
    elif dist['dist'] == 'uniform':
        lb = dist['lb']
        ub = dist['ub']
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    else:
        raise AssertionError("backlog function is not implemented.")
    
    data = data.astype(int)
    if save_data:
        save_array(data, f"env/{config_name}/backlogs.npy")
    return data
    

def generate_init_inventories(dist: dict, num_data: int, config_name: str="test", save_data: bool=True):
    
    if dist['dist'] == "constant":
        mean = dist['mean']
        data = np.array([mean for _ in range(num_data)]).astype(int)
    elif dist['dist'] == 'uniform':
        data = np.random.uniform(low=dist['lb'], high=dist['ub'], size=num_data).astype(int)
    else:
        raise AssertionError("init inventories is not implemented")
    
    if save_data:
        save_array(data, f"env/{config_name}/init_inventories.npy")
    return data

class Demand_fn:

    def __init__(self, dist: dict):
        # assert len(dist) == 3 if dist[0] == 'normal_demand' else 1, "Please provide the mean and std for the normal distribution."
        # assert len(dist) == 3 if dist[0] == 'uniform_demand' else 1, "Please provide the lower bound and upper bound for the uniform distribution."
        # assert len(dist) == 2 if dist[0] == 'constant_demand' else 1, "Please provide the mean value for the constant distribution."
        # assert len(dist) == 2 if "poisson_demand" in dist[0] else 1, "Please provide the mean value for the poisson distribution."
        
        self.lb = dist.get('lb', None)
        self.ub = dist.get('ub', None)
        self.mean = dist.get('mean', None)
        self.std = dist.get('std', None)
        self.dist = dist['dist']
        self.period = -1

        # Whether there is a random noise on demand
        if dist['with_noise']:
            self.noise = lambda x: np.random.poisson(lam=3)
        else: # poisson distribution of noise
            self.noise = lambda x: 0

        # There is a trend along time
        if dist['trend'] == "":
            self.trend = lambda t: 0 
        elif dist['trend'] == "linear":
            self.trend = lambda t: 2 * (t//2) 
        else: # TO-DO: other forms of trend
            raise ValueError("Trend function is not implemented")
 

    def constant_demand(self):
        return self.mean + self.trend(self.period) + self.noise(self.period)

    def uniform_demand(self):
        return np.random.randint(low=self.lb, high=self.ub) + self.trend(self.period) + self.noise(self.period)
    
    def normal_demand(self):
        return np.random.normal(self.mean, self.std) + self.trend(self.period) + self.noise(self.period)
    
    def poisson_demand(self):
        return np.random.poisson(self.mean) + self.trend(self.period) + self.noise(self.period)
    
    def seasonal_demand(self):
        return 3 * np.sin(2 * np.pi * 0.2 * self.period) + 5 + self.trend(self.period) + self.noise(self.period)
    
        
    def __call__(self, t):
        self.period = t
        if self.dist == 'constant_demand':
            return self.constant_demand()
        elif self.dist == "uniform_demand":
            return self.uniform_demand()
        elif self.dist == "normal_demand":
            return self.normal_demand()
        elif self.dist == "poisson_demand":
            return self.poisson_demand()
        elif self.dist == "seasonal_demand":
            return self.seasonal_demand()
        else:
            raise AssertionError("Demand function is not implemented.")
        
