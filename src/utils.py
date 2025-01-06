import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

def save_array(data: np.ndarray, save_path: str):
    print("Saving data to: ", save_path)
    np.save(save_path, data)

def convert_to_dict(input_string):
    # Remove unwanted characters and split the string
    input_string = input_string.replace('(', '').replace(')', '')
    pairs = input_string.split(', ')
    
    result_dict = {}
    
    for pair in pairs:
        # Split each pair into key and value
        key, value = pair.split(': ')
        result_dict[key.strip().strip('\"')] = int(value.strip())
        
    return result_dict

# Create a multipartite graph
def draw_multipartite_graph(env, t: int, save_prefix: str):

    num_stages = env.num_stages
    num_agents_per_stage = env.num_agents_per_stage
    sup_rel = env.supply_relations
    dem_rel = env.demand_relations
    save_path = f'results/{save_prefix}/'

    M = nx.DiGraph()

    # Add nodes for each set
    stage_agents = []
    for m in range(num_stages):
        stage_agents = []
        for x in range(num_agents_per_stage):
            stage_agents.append(f"s{m}a{x}")
        M.add_nodes_from(stage_agents, layer=num_stages-m)  # Add set A nodes


    # Add edges between the sets
    edges = []
    for m in range(num_stages-1):
        for x in range(num_agents_per_stage):
            for i in range(num_agents_per_stage):
                if sup_rel[m][x][i] == 1:
                    src = f"s{m+1}a{i}"
                    tgt = f"s{m}a{x}"
                    edges.append((src, tgt))

    M.add_edges_from(edges)

    # Define positions for the multipartite layout
    pos = nx.multipartite_layout(M, subset_key="layer")

    # Draw the multipartite graph
    # stage_colors = plt.cm.plasma(np.linspace(0, 1, 4))
    stage_colors = ["gold", "violet", "limegreen", "darkorange",]
    colors = [stage_colors[m] for m in range(num_stages) for x in range(num_agents_per_stage)]

    plt.figure(figsize=(10, 8))
    nx.draw(M, pos, with_labels=True, node_color=colors, node_size=2500, font_size=12, edge_color="gray", alpha=1)
    plt.title("Multipartite Graph")
    plt.savefig(os.path.join(save_path, f"supply_chain_period_{t}.jpg"), format="jpg")


def visualize_state(env, rewards: dict, t: int, save_prefix: str):
    
    state_dict = env.state_dict
    num_stages = env.num_stages
    num_agents_per_stage = env.num_agents_per_stage
    lt_max = env.max_lead_time
    save_path = f'results/{save_prefix}/'
    df = pd.DataFrame({
        "stage": {},
        "agent_idx": {}, 
        "prod_capacity": {},
        "sales_price": {},
        "order_cost": {},
        "backlog_cost": {},
        "holding_cost": {},
        "lead_time": {},
        "inventory": {},
        "backlog": {}, 
        "upstream_backlog": {},
        "recent_sales": {},
        "deliveries": {},
        "suppliers": {},
        "customers": {},
        "profits": {}
    })
    for stage in range(num_stages):
        for agent in range(num_agents_per_stage):

            df = pd.concat([df, pd.DataFrame({
                    'stage': [stage], 
                    "agent_idx": [agent],
                    "prod_capacity": [state_dict[f'stage_{stage}_agent_{agent}'][0]],
                    'sales_price': [state_dict[f'stage_{stage}_agent_{agent}'][1]],
                    'order_cost': [state_dict[f'stage_{stage}_agent_{agent}'][2]],
                    'backlog_cost': [state_dict[f'stage_{stage}_agent_{agent}'][3]],
                    'holding_cost': [state_dict[f'stage_{stage}_agent_{agent}'][4]],
                    'lead_time': [state_dict[f'stage_{stage}_agent_{agent}'][5]],
                    'inventory': [state_dict[f'stage_{stage}_agent_{agent}'][6]],
                    'backlog': [state_dict[f'stage_{stage}_agent_{agent}'][7]],
                    'upstream_backlog': [state_dict[f'stage_{stage}_agent_{agent}'][8]],
                    "suppliers": [state_dict[f'stage_{stage}_agent_{agent}'][9:9+num_agents_per_stage]],
                    "customers": [state_dict[f'stage_{stage}_agent_{agent}'][9+num_agents_per_stage:9+2*num_agents_per_stage]],
                    'recent_sales': [state_dict[f'stage_{stage}_agent_{agent}'][(-2 * lt_max):(-lt_max)].tolist()],
                    'deliveries': [state_dict[f'stage_{stage}_agent_{agent}'][-lt_max:].tolist()],
                    'profits': [rewards.get(f'stage_{stage}_agent_{agent}', None)]
                    })], ignore_index=True)
            
    df = df.groupby(by=['stage', 'agent_idx']).apply(lambda x: x).reset_index(drop=True)

    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, f"env_period_{t}.csv"), index=False)
    draw_multipartite_graph(env=env, t=t, save_prefix=save_prefix)


def random_relations(n_cand: int, n_relation: int):

    return np.random.choice(a=np.arange(n_cand), p=n_relation, replace=False)

def generate_lead_time(dist: str, num_data: int, num_agents_per_stage: int, lb=2, ub=8, config_name: str="test"):
    # To generate lead time for each agent
    if dist == 'uniform':
        data = np.random.uniform(low=lb, high=ub, size=(num_data, num_agents_per_stage))
    elif dist == "constant":
        mean = (lb + ub)//2
        data = [mean for _ in range(num_data)]
    else:
        raise AssertionError("Lead time function is not implemented.")
    save_array(data, f"../results/{config_name}/lead_time.npy")
    return data

def generate_prod_capacity(dist: str, num_data: int, lb: int=20, ub: int=40, config_name: str="test"):
    # To generate production capacity for each agent
    if dist == 'uniform':
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    elif dist == 'constant':
        mean = (lb + ub)//2
        data = [mean for _ in range(num_data)]
    else:
        raise AssertionError("Prod capacity function is not implemented.")
    
    save_array(data, f"../results/{config_name}/prod_capacity.npy")
    return data


def generate_profit_rate(dist: str, num_data: int, lb=0, ub=1, config_name: str="test"):
    # To generate profit rate for agents to decide price based on cost
    if dist == "uniform":
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    elif dist == 'constant':
        mean = (lb + ub)//2
        data = [mean for _ in range(num_data)]
    else:
        raise AssertionError("Profit rate function is not implemented.")
    
    save_array(data, f"../results/{config_name}/profit_rate.npy")
    return data

def generate_prod_cost(dist: str, num_data: int, lb=10, ub=20, config_name: str="test"):

    if dist == "uniform":
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    elif dist == "constant":
        mean = (lb + ub)//2
        data = [mean for _ in range(num_data)]
    else:
        raise AssertionError("Prod cost function is not implemented.")
    
    save_array(data, f"../results/{config_name}/prod_cost.npy")
    return data

def generate_cost_price(dist: str, num_stages: int, num_agents_per_stage: int, config_name: str="test"):

    # price = total cost * profit rate
    # cost = order cost + production cost
    num_total_agents = num_stages * num_agents_per_stage

    all_profit_rate = generate_profit_rate(dist=dist, num_data=num_total_agents)
    all_prod_costs = generate_prod_cost(dist=dist, num_data=num_total_agents)

    all_sale_prices = []
    all_total_costs = []

    manu_prices = all_prod_costs[:num_agents_per_stage] * all_profit_rate[:num_agents_per_stage]
    all_sale_prices += manu_prices # add prices of manufacturers to the price list
    all_total_costs += all_prod_costs[:num_agents_per_stage] # add cost of manufacturers to the cost list
    for i in range(1, num_stages):
        order_costs = all_sale_prices[-num_agents_per_stage:]
        prod_costs = all_prod_costs[i*num_agents_per_stage:(i+1)*num_agents_per_stage]
        profit_rate = all_profit_rate[i*num_agents_per_stage:(i+1)*num_agents_per_stage]
        downstream_prices = (order_costs + prod_costs) * profit_rate

        all_sale_prices += downstream_prices
    
    save_array(all_sale_prices, f"../results/{config_name}/sale_prices.npy")
    save_array(all_total_costs, f"../results/{config_name}/total_costs.npy")
    return all_total_costs, all_sale_prices


def generate_sup_dem_relations(type: str, num_stages: int, num_agents_per_stage: int):

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
    else:
        raise AssertionError("Relation function is not implemented.")
    

def generate_holding_costs(dist: str, num_data: int, lb: int=1, ub: int=5, config_name: str="test"):

    if dist == 'constant':
        mean = (lb + ub)//2
        data = [mean for _ in range(num_data)]
    elif dist == "uniform":
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    else:
        raise AssertionError("holding function is not implemented.")

    save_array(data, f"../results/{config_name}/holding_costs.npy")
    return data


def generate_backlog_costs(dist: str, num_data: int, lb: int=1, ub: int=5, config_name: str="test"):

    if dist == 'constant':
        mean = (lb + ub)//2
        data = [mean for _ in range(num_data)]
    elif dist == "uniform":
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    else:
        raise AssertionError("backlog function is not implemented.")
    
    save_array(data, f"../results/{config_name}/backlog_costs.npy")
    return data
    

def generate_init_inventories(dist: str, num_data: int, lb: int=10, ub: int=18, config_name: str="test"):

    if dist == "constant":
        mean = (lb+ub)//2
        data = [mean for _ in range(num_data)]
    elif dist == 'uniform':
        data = np.random.uniform(low=lb, high=ub, size=num_data)
    else:
        raise AssertionError("init inventories is not implemented")
    
    save_array(data, f"../results/{config_name}/init_inventories.npy")
    return data

    

class Demand_fn:

    def __init__(self, dist: str, lb: int=2, ub: int=8, mean: int=4):
        self.dist = dist
        self.lb = lb
        self.ub = ub
        self.mean = mean
        self.period = -1

    def constant_demand(self):
        return self.mean

    def uniform_demand(self):
        return np.random.randint(low=self.lb, high=self.ub)
    
    def forward(self, t):
        self.period = t
        if self.dist == 'constant':
            return self.constant_demand()
        elif self.dist == "uniform":
            return self.uniform_demand()