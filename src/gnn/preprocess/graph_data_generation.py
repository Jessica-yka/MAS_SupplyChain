# %% [markdown]
# ## Graph Data Generation
# 1. Design context pool (event and edge)
# 1. Randomly generate graph
#     * 2 relations: 1000 data
#     * 3 relations: 1000 data
#     * 4 relations: 500 data
# * Randomly assign events to nodes
# * generate label for two types of question (suppliers reliability/downstream demand)
# * generate label

# %%
import numpy as np
import pandas as pd
import os
import re
import sys
import time
sys.path.append('/data/yanjia/MAS_SupplyChain')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.config import env_configs_list
from src.model.utils.utils import clear_dir, split_demand, save_data_to_json, read_data_from_json
from src.model.data_simulation import generate_lead_time, generate_prod_capacity, generate_backlogs
from src.model.data_simulation import generate_cost_price, generate_sup_dem_relations
from src.model.data_simulation import generate_holding_costs, generate_backlog_costs, generate_init_inventories
from src.model.data_simulation import Demand_fn
import matplotlib.pyplot as plt
import networkx as nx
import random
import csv
from tqdm import tqdm
import torch
from scipy.stats import rankdata


# %% [markdown]
# ## Event Pool

# %%

# Define the list of events with simplified descriptions
events = [
    ["Event", "Effect Type", "Affected Aspect"],
    ["Earthquakes", "Negative", ["Production Capacity"]],
    ["Hurricanes", "Negative", ["Delivery Time"]],
    ["Floods", "Negative", ["Order Fulfillment"]],
    ["Wildfires", "Negative", ["Production Capacity"]],
    ["Droughts", "Negative", ["Price"]],
    ["Tsunamis", "Negative", ["Delivery Time"]],
    ["Volcano eruptions", "Negative", ["Delivery Time"]],
    ["Severe storms", "Negative", ["Delivery Time"]],
    ["Pandemics", "Negative", ["Production Capacity"]],
    ["Factory closures", "Negative", ["Production Capacity"]],
    ["Workforce absenteeism", "Negative", ["Production Capacity"]],
    ["Recessions", "Negative", ["Demand"]],
    ["Inflation", "Negative", ["Price"]],
    ["Trade wars", "Negative", ["Price"]],
    ["Economic booms", "Positive", ["Demand"]],
    ["Wage increases", "Negative", ["Price"]],
    ["Labor strikes", "Negative", ["Production Capacity"]],
    ["Port congestion", "Negative", ["Delivery Time"]],
    ["Fuel shortages", "Negative", ["Delivery Time"]],
    ["Road closures", "Negative", ["Delivery Time"]],
    ["Truck driver shortages", "Negative", ["Delivery Time"]],
    ["Rail strikes", "Negative", ["Delivery Time"]],
    ["Air traffic disruptions", "Negative", ["Delivery Time"]],
    ["Shipping container shortages", "Negative", ["Price"]],
    ["New environmental regulations", "Negative", ["Price"]],
    ["Bans on certain materials", "Negative", ["Production Capacity"]],
    ["Import/export law changes", "Negative", ["Delivery Time"]],
    ["Tax reforms", "Positive", ["Price"]],
    ["Licensing delays", "Negative", ["Production Capacity"]],
    ["Raw material shortages", "Negative", ["Price"]],
    ["Discovery of abundant raw materials", "Positive", ["Price"]],
    ["Mining accidents", "Negative", ["Production Capacity"]],
    ["Water scarcity", "Negative", ["Price"]],
    ["Automation", "Positive", ["Production Capacity"]],
    ["AI implementation", "Positive", ["Production Capacity"]],
    ["Better forecasting tools", "Positive", ["Order Fulfillment"]],
    ["Drone delivery", "Positive", ["Delivery Time"]],
    ["Renewable energy adoption", "Positive", ["Price"]],
    ["Consumer preference shifts", "Positive", ["Demand"]],
    ["Seasonal demand spikes", "Positive", ["Demand"]],
    ["New product version launches", "Positive", ["Demand"]],
    ["Social media campaigns", "Positive", ["Demand"]],
    ["Negative publicity", "Negative", ["Demand"]],
    ["War", "Negative", ["Delivery Time"]],
    ["Sanctions", "Negative", ["Production Capacity"]],
    ["Political instability", "Negative", ["Delivery Time"]],
    ["Trade agreements", "Positive", ["Price"]],
    ["Border closures", "Negative", ["Delivery Time"]],
    ["Factory fires", "Negative", ["Production Capacity"]],
    ["Equipment breakdowns", "Negative", ["Production Capacity"]],
    ["Lean manufacturing", "Positive", ["Production Capacity"]],
    ["Outsourcing", "Positive", ["Price"]],
    ["Climate change", "Negative", ["Price"]],
    ["Extreme heat", "Negative", ["Production Capacity"]],
    ["Biodiversity loss", "Negative", ["Production Capacity"]],
    ["Panic buying", "Positive", ["Demand"]],
    ["Increased popularity", "Positive", ["Demand"]],
    ["New substitute", "Negative", ["Demand"]],
    ["Viral trends", "Positive", ["Demand"]],
    ["Cyberattacks", "Negative", ["Production Capacity"]],
    ["Supply chain software failures", "Negative", ["Delivery Time"]],
    ["Data breaches", "Negative", ["Demand"]],
    ["Blockchain implementation", "Positive", ["Production Capacity", "Delivery Time"]],
    ["Rising oil prices", "Negative", ["Price"]],
    ["Power outages", "Negative", ["Production Capacity"]],
    ["Electrification of fleets", "Positive", ["Price", "Delivery Time"]],
    ["Just-in-Time (JIT) implementation", "Positive", ["Order Fulfillment"]],
    ["Customer loss due to delays", "Negative", ["Demand"]],
    ["Premium pricing for faster delivery", "Positive", ["Price"]],
    ["Trade show cancellations", "Negative", ["Demand"]],
    ["Major sporting events", "Positive", ["Demand"]],
    ["Packaging shortages", "Negative", ["Production Capacity", "Delivery Time"]],
    ["Transportation accidents", "Negative", ["Delivery Time"]],
    ["New packaging", "Positive", ["Price"]],
    ["Market booms in emerging economies", "Positive", ["Demand"]],
    ["Subsidies", "Positive", ["Price"]],
    ["Higher taxes", "Negative", ["Price"]],
    ["Mergers", "Positive", ["Delivery Time"]],
    ["Natural gas shortages", "Negative", ["Production Capacity", "Price"]],
    ["Government incentives for green energy", "Positive", ["Production Capacity"]],
    ["Adoption of electric vehicles in logistics", "Positive", ["Delivery Time", "Price"]],
    ["Breakthroughs in recycling technology", "Positive", ["Order Fulfillment", "Price"]],
    ["Government subsidies for local manufacturing", "Positive", ["Production Capacity", "Price"]],
    ["Introduction of autonomous delivery systems", "Positive", ["Delivery Time"]],
    ["High investor trust in brand reliability", "Positive", ["Demand"]],
    ["Partnerships with local suppliers", "Positive", ["Delivery Time", "Production Capacity"]],
    ["Development of predictive maintenance systems", "Positive", ["Production Capacity"]],
    ["Global reduction in trade tariffs", "Positive", ["Price", "Delivery Time"]],
    ["Increased investment in renewable energy infrastructure", "Positive", ["Production Capacity", "Price"]],
    ["Advances in operation robotics", "Positive", ["Production Capacity"]]
]



save_data_to_json(data=events, save_path="src/gnn/gnn_dataset/supply_chain_events.json")

print("CSV file 'supply_chain_events.csv' created successfully!")

# %% [markdown]
# ## Environment generation

# %%
    
def visualize_contextualized_supply_chain(env: dict, event_dict: dict, df_edges: pd.DataFrame, df_nodes: pd.DataFrame, path: str):
    
    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    stage_name_id = dict(zip(env['stage_names'], range(num_stages)))
    M = nx.DiGraph()

    # Add nodes for each set
    for i in range(len(df_nodes)):
        if df_nodes['type'][i] == "event":
            M.add_node(df_nodes['node_attr'][i], type=df_nodes['type'][i])
        else:
            M.add_node(df_nodes['node_attr'][i], type=num_stages-1-stage_name_id[df_nodes['type'][i]])

    # Add edges between the sets
    for i in range(len(df_edges)):
        M.add_edge(df_edges['source'][i], df_edges['target'][i], label=df_edges['label'][i])

    # Define positions for the multipartite layout
    pos = nx.multipartite_layout(M, subset_key="type")
    edge_labels = nx.get_edge_attributes(M, "label") # Get edge labels
    # Draw the multipartite graph
    # stage_colors = plt.cm.plasma(np.linspace(0, 1, 4))
    stage_colors = {0: "gold", 1: "violet", 2: "limegreen", 3:"darkorange", "event": "blue"}
    colors = [stage_colors[m.get("type")] for m in M.nodes.values()]


    plt.figure(figsize=(20, 16))
    nx.draw(M, pos, with_labels=True, node_color=colors, node_size=1000, font_size=12, edge_color="gray", alpha=1)
    nx.draw_networkx_edge_labels(M, pos, edge_labels=edge_labels, font_size=10)
    # plt.show()
    plt.savefig(path)


def visualize_contextualized_supply_chain_subgraph(env: dict, event_dict: dict, df_edges: pd.DataFrame, df_nodes: pd.DataFrame, target_node: str, path: str):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    stage_name_id = dict(zip(env['stage_names'], range(num_stages)))
    M = nx.DiGraph()

    # Add all nodes to the graph
    for i in range(len(df_nodes)):
        if df_nodes['type'][i] == "event":
            M.add_node(df_nodes['name'][i], type="event")
        else:
            M.add_node(df_nodes['name'][i], type=num_stages-1-stage_name_id[df_nodes['type'][i]])

    # Add edges to the graph if
    # supply relation nodes
    # the deliverying and ordering between the target and its downstream/upstream
    for i in range(len(df_edges)):
        source = df_edges['src_name'][i]
        target = df_edges['dst_name'][i]
        label = df_edges['edge_attr'][i]
        M.add_edge(source, target, label=label)


    # Define positions for the multipartite layout
    pos = nx.multipartite_layout(M, subset_key="type")
    edge_labels = nx.get_edge_attributes(M, "label") # Get edge labels
    # Draw the multipartite graph
    # stage_colors = plt.cm.plasma(np.linspace(0, 1, 4))
    stage_colors = {0: "gold", 1: "violet", 2: "limegreen", 3:"darkorange", "event": "blue"}
    colors = [stage_colors[m.get("type")] for m in M.nodes.values()]


    plt.figure(figsize=(15, 12))
    nx.draw(M, pos, with_labels=True, node_color=colors, node_size=1000, font_size=12, edge_color="gray", alpha=1)
    nx.draw_networkx_edge_labels(M, pos, edge_labels=edge_labels, font_size=10)
    # plt.show()
    plt.savefig(path)
    plt.close()


def assign_events(num_events: int, num_stages: int, num_agents_per_stage: int):

    # num_current_events = random.choice(range(1, 4))
    num_current_events = 1
    event_idx = random.sample(range(num_events), num_current_events)
    assigned_agents = []
    for _ in range(num_current_events):
        stage_idx = random.choice(range(num_stages))
        agent_idx = random.choice(range(num_agents_per_stage))
        assigned_agents.append((stage_idx, agent_idx))
    return dict(zip(event_idx, assigned_agents))

# %%
def convert_env_to_node_df(env: dict):
    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    stage_names = env['stage_names']
    num_current_events = len(env['events'])
    num_nodes = num_stages * num_agents_per_stage + num_current_events
    df_node = pd.DataFrame(index=range(num_nodes), columns=["node_id", "type"])
    df_node["node_id"] = np.arange(num_nodes)
    df_node["name"] = [f"stage_{m}_agent_{x}" for m in range(num_stages) for x in range(num_agents_per_stage)] + [event_dict['events'][eidx] for eidx in env['events'].keys()]
    df_node["type"] = [stage_names[m] for m in range(num_stages) for x in range(num_agents_per_stage)] + ["event" for _ in range(num_current_events)]
    df_node['sale_price'] = env['sale_prices'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['prod_capacity'] = env['prod_capacities'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['prod_cost'] = env['prod_costs'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['holding_cost'] = env['holding_costs'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['backlog_cost'] = env['backlog_costs'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['inventory'] = env['inventories'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['backlog'] = env['backlog'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['upstream_backlog'] = [0 for _ in range(num_stages*num_agents_per_stage)] + [0 for _ in range(num_current_events)]
    df_node['stage_id'] = [m for m in range(num_stages) for _ in range(num_agents_per_stage)] + [-1 for _ in env['events'].keys()]
    df_node['agent_id'] = [x for _ in range(num_stages) for x in range(num_agents_per_stage)] + [-1 for _ in env['events'].keys()]
    
    return df_node


# %%
def convert_env_to_edge_df(env: dict, event_dict: dict):
    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    num_init_suppliers = env['num_init_suppliers']
    sup_rel = env['supply_relations']
    order_fulfill_rates = env['order_fulfill_rates']
    num_edges = sum([sum([sum(sup_rel[m][x]) for x in range(num_agents_per_stage)]) for m in range(num_stages-1)]) + len(env['events'])
    df_edge = pd.DataFrame(index=range(num_edges), columns=["source", "target", "label", "type", 'aspect'])
    edge_idx = 0
    t = random.choice(range(1, env['num_periods']))
    # Randomly create backlog events between suppliers and customers
    for m in range(num_stages-1):
        for x in range(num_agents_per_stage):
            for i in range(num_agents_per_stage):
                if sup_rel[m][x][i] == 1:
                    num_request_order = env['demand_fn'](t)//num_init_suppliers
                    num_fufilled_order = int(num_request_order * order_fulfill_rates[m+1][i][x]) # the fulfillment rate of the supplier stage_(m+1)_agent_i to the customer stage_m_agent_x
                    is_fulfilled = "Positive" if num_fufilled_order >= num_request_order else "Negative"
                    df_edge.loc[edge_idx, ["source", "target", "label", "type", 'aspect']] = \
                        [f"stage_{m}_agent_{x}", f"stage_{m+1}_agent_{i}", f"request order of {num_request_order} units of product at round {t-1}", "", []]
                    df_edge.loc[edge_idx+1, ["source", "target", "label", 'type', 'aspect']] = \
                        [f"stage_{m+1}_agent_{i}", f"stage_{m}_agent_{x}", f"deliverying {num_fufilled_order} units of product at round {t}", is_fulfilled, ['Order Fulfillment']]
                    edge_idx += 2

    # Keep the record of the other events
    for eidx in env['events'].keys():
        event_name = event_dict['events'][eidx]
        stage_idx, agent_idx = env['events'][eidx]
        event_type = event_dict['Type'][eidx]
        aspect = event_dict['Aspect'][eidx]
        if event_type == 'Postive':
            df_edge.loc[edge_idx, ["source", "target", "label", 'type', 'aspect']] = \
                [event_name, f"stage_{stage_idx}_agent_{agent_idx}", "positively affects", event_type, aspect]
        else: # event_type == 'Negative'
            df_edge.loc[edge_idx, ["source", "target", "label", 'type', 'aspect']] = \
                [event_name, f"stage_{stage_idx}_agent_{agent_idx}", "negatively affects", event_type, aspect]
        edge_idx += 1

    # Keep the record of the supply relations
    for m in range(num_stages-1):
        for x in range(num_agents_per_stage):
            for i in range(num_agents_per_stage):
                if sup_rel[m][x][i] == 1:
                    df_edge.loc[edge_idx, ["source", "target", "label", 'type', 'aspect']] = \
                        [f"stage_{m+1}_agent_{i}", f"stage_{m}_agent_{x}", "is the supplier of", "", []]
                    edge_idx += 1
                else:
                    pass
    
    return df_edge


def build_supplier_graph(df_edges: pd.DataFrame, df_nodes: pd.DataFrame):

    G = nx.DiGraph()
    for i in range(len(df_nodes)):
        G.add_node(df_nodes['name'][i], type=df_nodes['type'][i])
    for i in range(len(df_edges)):
        label = df_edges['label'][i]
        if label == 'is the supplier of':
            G.add_edge(df_edges['source'][i], df_edges['target'][i], label=label, type=df_edges['type'][i], aspect=df_edges['aspect'][i])
    return G


def generate_orderFulfill_rates(supply_relations: np.array, num_stages: int, num_agents_per_stage: int):

    order_fulfill_rates = np.zeros((num_stages, num_agents_per_stage, num_agents_per_stage))
    for m in range(num_stages-1):
        for x in range(num_agents_per_stage):
            for i in range(num_agents_per_stage):
                if supply_relations[m][x][i] == 1:
                    order_fulfill_rates[m+1][i][x] = min(1, np.random.uniform(0.5, 1.5))

    return order_fulfill_rates


def generate_env(env_config_name: str):

    env_configs = env_configs_list[env_config_name]
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
                            num_stages=num_stages, num_agents_per_stage=num_agents_per_stage, config_name=env_configs["config_name"], save_data=False)
    holding_costs = \
        generate_holding_costs(dist=env_configs["holding_costs_dist"], num_data=num_total_agents, config_name=env_configs["config_name"], save_data=False)
    backlog_costs = \
        generate_backlog_costs(dist=env_configs["backlog_costs_dist"], num_data=num_total_agents, config_name=env_configs["config_name"], save_data=False)
    lead_times = \
        generate_lead_time(dist=env_configs["lead_time_dist"], num_stages=num_stages, num_agents_per_stage=num_agents_per_stage,config_name=env_configs["config_name"], save_data=False)
    prod_capacities = \
        generate_prod_capacity(dist=env_configs['prod_capacity_dist'], num_data=num_total_agents, config_name=env_configs["config_name"], save_data=False)
    init_inventories = \
        generate_init_inventories(dist=env_configs["init_inventory_dist"], num_data=num_total_agents, config_name=env_configs["config_name"], save_data=False)
    backlogs = \
        generate_backlogs(dist={'dist': 'uniform', 'lb': 0, 'ub': 5}, num_data=num_total_agents, config_name=env_configs["config_name"], save_data=False)
    # profit_rates = \
    #     generate_profit_rates(dist=env_configs["profit_rate_dist"], num_data=num_total_agents, config_name=env_configs["config_name"])
    order_fulfill_rates = generate_orderFulfill_rates(supply_relations=supply_relations, num_stages=num_stages, num_agents_per_stage=num_agents_per_stage)

    demand_fn = Demand_fn(dist=env_configs["demand_fn"])
    stage_names = env_configs["stage_names"]

    return {
            'num_stages': num_stages,
            'num_periods': num_periods,
            't': random.choice(range(num_periods-1)),
            'num_agents_per_stage': num_agents_per_stage,
            "demand_dist": env_configs["demand_fn"]["dist"],
            'inventories': init_inventories, # num_stages * num_agents_per_stage
            'lead_times': lead_times, # num_stages * num_agents_per_stage * num_agents_per_stage
            'demand_fn': demand_fn,
            'prod_capacities': prod_capacities,
            'sale_prices': sale_prices,
            'order_costs': order_costs,
            "prod_costs": prod_costs, 
            'backlog_costs': backlog_costs,
            'backlog': backlogs,
            'holding_costs': holding_costs,
            'num_init_suppliers': num_init_suppliers, 
            'supply_relations': supply_relations,
            "demand_relations": demand_relations,
            'stage_names': stage_names,
            'order_fulfill_rates': order_fulfill_rates,
        }

# %%
## Randomly generate questions
# 1. assign the target node
# 2. Question: Whether the event would affect the target node positively or negatively in terms of the aspect
# 3. Search the graph to see if there is a path from the event to the target node (omit the direction, just check the connection)
# 4. If there is a path, the answer is  positive/negative (based on the aspect)
# 5. If there is no path, the answer is "neutral"
# 6. Generate 10 questions
def check_connection(G, event_target_node, target_node):
    return nx.has_path(G, source=event_target_node, target=target_node)

def generate_target_node(num_stages: int, num_agents_per_stage: int):

    target_node_stage_id = random.choice(range(num_stages))
    target_node_agent_id = random.choice(range(num_agents_per_stage))

    return target_node_stage_id, target_node_agent_id

def generate_orderFulfill_questions(num_periods: int, target_node:str, event_node:str, event_type: str):

    t = env['t']
    question = f"Your are {target_node} at round {t}. Based on the provided supply chain graph, how is the performance of your supplier {event_node} in terms of order fulfillment? Answer either 'positive' or 'negative'."
    answer = "positive" if event_type == "Positive" else "negative"

    return question, answer

def get_event_sub_df_edges(G: nx.DiGraph, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, target_node: str, path: str=None):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']

    G_sub = G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True) if d['label']=='is the supplier of']).copy()

    related_nodes = list(nx.nodes(nx.dfs_tree(G_sub, target_node))) + list(nx.nodes(nx.dfs_tree(G_sub.reverse(), target_node)))
    related_nodes = dict(zip(related_nodes, [1 for _ in related_nodes]))

    node_name_id_map = dict(zip(df_nodes['name'].tolist(), df_nodes['node_id'].tolist()))
    df_simp_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
    
    row_idx = 0
    # list all the edges in G_sub and make it df_simp_edges
    # get the label of edges
    for src_name, dst_name, data in G_sub.edges(data=True):
        if related_nodes.get(src_name, 0) and related_nodes.get(dst_name, 0):
            df_simp_edges.loc[row_idx] = [node_name_id_map[src_name], data['label'], node_name_id_map[dst_name], src_name, dst_name]
            row_idx += 1
       
    # add lead time info to the edge_df
    # _, stage_id, _, agent_id = target_node.split('_')
    # stage_id = int(stage_id)
    # agent_id = int(agent_id)

    # if stage_id < num_stages-1:
    #     for i in range(num_agents_per_stage):
    #         lt = env['lead_times'][stage_id][agent_id][i]
    #         df_simp_edges.loc[row_idx] = [node_name_id_map[f"stage_{stage_id+1}_agent_{i}"], 
    #                                       f"has lead time of {lt} days to", node_name_id_map[target_node], 
    #                                       f"stage_{stage_id+1}_agent_{i}", target_node]
    #         row_idx += 1
    
    df_events = df_edges['affects' in df_edges['label']].reset_index(drop=True)
    for i in range(len(df_events)):
        df_simp_edges.loc[row_idx] = [node_name_id_map[df_events.loc[i, 'source']], df_events.loc[i, 'label'], node_name_id_map[df_events.loc[i, 'target']], df_events.loc[i, 'source'], df_events.loc[i, 'target']]
        row_idx += 1

    # # add order info to the edge_df
    # df_self= df_edges[(df_edges['source'] == target_node) | (df_edges['target']==target_node)].reset_index(drop=True)
    # for i in range(len(df_self)):
    #     df_simp_edges.loc[row_idx] = [node_name_id_map[df_self.loc[i, 'source']], df_self.loc[i, 'label'], node_name_id_map[df_self.loc[i, 'target']], df_self.loc[i, 'source'], df_self.loc[i, 'target']]
    #     row_idx += 1    

    return df_simp_edges

def get_supplier_sub_df_edges(G: nx.DiGraph, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, target_node: str, path: str=None):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']

    G_sub = G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True) if d['label']=='is the supplier of']).copy()

    related_nodes = list(nx.nodes(nx.dfs_tree(G_sub, target_node))) + list(nx.nodes(nx.dfs_tree(G_sub.reverse(), target_node)))
    related_nodes = dict(zip(related_nodes, [1 for _ in related_nodes]))

    node_name_id_map = dict(zip(df_nodes['name'].tolist(), df_nodes['node_id'].tolist()))
    df_simp_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
    
    row_idx = 0
    # list all the edges in G_sub and make it df_simp_edges
    # get the label of edges
    for src_name, dst_name, data in G_sub.edges(data=True):
        if related_nodes.get(src_name, 0) and related_nodes.get(dst_name, 0):
            df_simp_edges.loc[row_idx] = [node_name_id_map[src_name], data['label'], node_name_id_map[dst_name], src_name, dst_name]
            row_idx += 1
       

    return df_simp_edges


def get_lt_sub_df_edges(G: nx.DiGraph, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, target_node: str, path: str=None):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']

    node_name_id_map = dict(zip(df_nodes['name'].tolist(), df_nodes['node_id'].tolist()))
    df_simp_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
    
    row_idx = 0
       
    # add lead time info to the edge_df
    _, stage_id, _, agent_id = target_node.split('_')
    stage_id = int(stage_id)
    agent_id = int(agent_id)

    if stage_id < num_stages-1:
        for i in range(num_agents_per_stage):
            lt = env['lead_times'][stage_id][agent_id][i]
            df_simp_edges.loc[row_idx] = [node_name_id_map[f"stage_{stage_id+1}_agent_{i}"], 
                                          f"has lead time of {lt} days to", node_name_id_map[target_node], 
                                          f"stage_{stage_id+1}_agent_{i}", target_node]
            row_idx += 1

    return df_simp_edges

def get_price_sub_df_edges(G: nx.DiGraph, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, target_node: str, path: str=None):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']


    node_name_id_map = dict(zip(df_nodes['name'].tolist(), df_nodes['node_id'].tolist()))
    df_simp_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
    
    row_idx = 0

    # add potential suppliers info to the edge_df
    _, stage_id, _, agent_id = target_node.split('_')
    stage_id = int(stage_id)
    agent_id = int(agent_id)

    if stage_id < num_stages-1:
        for i in range(num_agents_per_stage):
            df_simp_edges.loc[row_idx] = [node_name_id_map[f"stage_{stage_id+1}_agent_{i}"], 
                                          f"is an upstream agent to", node_name_id_map[target_node], 
                                          f"stage_{stage_id+1}_agent_{i}", target_node]
            row_idx += 1
    

    return df_simp_edges

def get_sub_df_edges(G: nx.DiGraph, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, target_node: str, path: str=None):
    pass

def get_sub_df_nodes(df_nodes: pd.DataFrame, target_node: str, path: str=None):

    df_nodes_sub = pd.DataFrame(columns=['node_id', 'node_attr', 'type', 'name'])
    # the competitors at the save stage
    for i in range(len(df_nodes)):
        # itself
        if df_nodes.loc[i, 'name'] == target_node:
            attr = (f"{df_nodes.loc[i, 'name']}: "
                    f"price: {df_nodes.loc[i, 'sale_price']}, "
                    f"production cost: {df_nodes.loc[i, 'prod_cost']}, "
                    f"production capacity: {df_nodes.loc[i, 'prod_capacity']}, "
                    f"inventory: {df_nodes.loc[i, 'inventory']}, "
                    f"backlog: {df_nodes.loc[i, 'backlog']}, "
                    f"upstream backlog: {df_nodes.loc[i, 'upstream_backlog']}")
            df_nodes_sub.loc[i, ['node_id', 'node_attr', 'type', 'name']] = [df_nodes.loc[i, 'node_id'], attr, df_nodes.loc[i, 'type'], df_nodes.loc[i, 'name']]
        # the suppliers of the target node
        elif f"stage_{df_nodes.loc[i, 'stage_id']-1}" in target_node:
            attr = (f"{df_nodes.loc[i, 'name']}: "
                    f"price: {df_nodes.loc[i, 'sale_price']}, "
                    f"production capacity: {df_nodes.loc[i, 'prod_capacity']}")
            df_nodes_sub.loc[i, ['node_id', 'node_attr', 'type', 'name']] = [df_nodes.loc[i, 'node_id'], attr, df_nodes.loc[i, 'type'], df_nodes.loc[i, 'name']]
        else: # the suppliers of the suppliers or the downstream customers
            attr = (f"{df_nodes.loc[i, 'name']}")
            df_nodes_sub.loc[i, ['node_id', 'node_attr', 'type', 'name']] = [df_nodes.loc[i, 'node_id'], attr, df_nodes.loc[i, 'type'], df_nodes.loc[i, 'name']]

    # df_nodes_sub.to_csv(path, index=False)
    return df_nodes_sub


def list_all_successor_nodes(G, node):
    # select successor based on "is supplier of" relation only
    return [n for n in nx.nodes(nx.dfs_tree(G, node)) if n.split("_")[1] < node.split("_")[1]]

def list_all_predecessor_nodes(G, node):
    return [n for n in nx.nodes(nx.dfs_tree(G.reverse(), node)) if 'stage' in n and n.split("_")[1] > node.split("_")[1]]

def generate_event_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env: dict, data_idx:int, num_questions:int=10):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    t = env['t']
    aspect_list = ["Production Capacity", "Delivery Time", "Order Fulfillment", "Price", "Demand"]
    up_aspect_list = ["Production Capacity", "Delivery Time", "Order Fulfillment", "Price"]
    down_aspect_list = ["Demand"]

    # make a list of list to list
    event_aspect_in_graph = []
    for x in df_edges['aspect']:
        event_aspect_in_graph += x
    event_aspect_in_graph = list(set(event_aspect_in_graph))
    # event_aspect_in_graph + ["Order Fulfillment" for _ in event_aspect_in_graph] # To balance the question ratio
    num_event_aspect_in_graph = len(event_aspect_in_graph)
    
    # remove price from the aspect list
    questions = []
    answers = []

    n_cum_questions = 0
    while n_cum_questions < num_questions:
        try:
            # Get a valid event
            es = event_aspect_in_graph[n_cum_questions%num_event_aspect_in_graph]
            df_event_aspect = df_edges[df_edges['aspect'].apply(lambda x: es in x)].reset_index(drop=True)
            row_id = random.choice(range(len(df_event_aspect)))
            event_aspect = df_event_aspect.loc[row_id, 'aspect']

            event_type = df_event_aspect.loc[row_id, 'type']
            event_target_node = df_event_aspect.loc[row_id, 'target']
            event_node = df_event_aspect.loc[row_id, 'source']
            # Special case to deal with backlog judgement
            if "Order Fulfillment" in event_aspect:
                target_node=event_target_node
                question, answer = generate_orderFulfill_questions(num_periods=env['num_periods'], target_node=event_target_node, event_node=event_node, event_type=event_type)
            else:
                cases = random.choices(['not connected', 'downstream', 'upstream'], weights=[0.4,0.2,0.4], k=1)[0] # case happens to upstream/downstream/not connected
                if cases == 'not connected': # not connected
                    target_node_stage_id, target_node_agent_id = generate_target_node(num_stages=num_stages, num_agents_per_stage=num_agents_per_stage)
                    predecessors = list_all_predecessor_nodes(G, event_target_node)
                    successors = list_all_successor_nodes(G, event_target_node)
                    while f"stage_{target_node_stage_id}_agent_{target_node_agent_id}" in predecessors \
                        or f"stage_{target_node_stage_id}_agent_{target_node_agent_id}" in successors \
                        or (int(event_target_node.split("_")[1]) == target_node_stage_id and int(event_target_node.split("_")[3]) == target_node_agent_id):
                        target_node_stage_id, target_node_agent_id = generate_target_node(num_stages=num_stages, num_agents_per_stage=num_agents_per_stage)
                    target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"
                    
                    event_target_node_relation = random.choice(['upstream suppliers reliability', 'downstream customers'])
                    node_rel = 'suppliers' if event_target_node_relation == 'upstream suppliers reliability' else 'customers'

                    asp = random.choice(down_aspect_list) if node_rel == 'customers' else random.choice(up_aspect_list)
                    question = f"Your are {target_node} at round {t}. Based on the provided supply chain graph, how would the {event_node} affect your {event_target_node_relation} in terms of {asp}? Answer either 'positive' or 'negative' if it happens to your {node_rel}(s), otherwise answer 'neutral'."
                    answer = "neutral"

                elif cases == 'downstream':
                    asp = random.choice(down_aspect_list)
                    try:
                        target_node = random.choice(list_all_predecessor_nodes(G=G, node=event_target_node))
                    except:
                        continue
                    question = f"Your are {target_node} at round {t}. Based on the provided supply chain graph, how would the {event_node} affect your downstream customers in terms of {asp}? Answer either 'positive' or 'negative' if it happens to your customers(s), otherwise answer 'neutral'."
                    answer = "positive" if event_type == "Positive" else "negative"
                else:
                    asp = random.choice(up_aspect_list)
                    try:
                        target_node = random.choice(list_all_successor_nodes(G=G, node=event_target_node))
                    except:
                        continue
                    question = f"Your are {target_node} at round {t}. Based on the provided supply chain graph, how would the {event_node} affect your upstream suppliers reliability in terms of {asp}? Answer either 'positive' or 'negative' if it happens to your supplier(s), otherwise answer 'neutral'."
                    answer = "positive" if event_type == "Positive" else "negative"
            
            # Save the target-node-related graph as node df/edge df/graph/graph img
            df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node, path=f"{save_path}/{env_config_name}/nodes/{data_idx+n_cum_questions}.csv")
            df_simp_edges = get_event_sub_df_edges(G=G, df_nodes=df_sub_nodes, df_edges=df_edges, target_node=target_node, path=f"{save_path}/{env_config_name}/edges/{data_idx+n_cum_questions}.csv")
            visualize_contextualized_supply_chain_subgraph(env=env, event_dict=event_dict, target_node=target_node, df_edges=df_simp_edges, df_nodes=df_sub_nodes, path=f"{save_path}/{env_config_name}/graph_imgs/{data_idx+n_cum_questions}.png")
            df_sub_nodes.to_csv(f"{save_path}/{env_config_name}/nodes/{data_idx+n_cum_questions}.csv", index=False)
            df_simp_edges.to_csv(f"{save_path}/{env_config_name}/edges/{data_idx+n_cum_questions}.csv", index=False)
            questions.append(question)
            answers.append(answer)
            n_cum_questions += 1
        except:
            pass

    return questions, answers


def generate_supplier_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env: dict, data_idx:int, num_questions:int=10): 

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    t = env['t']
    questions = []
    answers = []

    for n_cum_questions in range(num_questions):
        target_node_stage_id = random.choice(range(num_stages-1))
        target_node_agent_id = random.choice(range(num_agents_per_stage))
        target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"
        supply_relations = env['supply_relations'][target_node_stage_id][target_node_agent_id]

        # Randomly select a supplier
        upstream_agent_id = random.choice(range(num_agents_per_stage))
        upstream_node = f"stage_{target_node_stage_id+1}_agent_{upstream_agent_id}"
        question = f"Your are {target_node} at round {t}. Based on the provided supply chain graph, is {upstream_node} one of your suppliers? Answer either 'yes' or 'no'."
        questions.append(question)
        if supply_relations[upstream_agent_id] == 1:
            answers.append("yes")
        else:
            answers.append("no")

        # Save the target-node-related graph as node df/edge df/graph/graph img
        df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node)
        df_simp_edges = get_supplier_sub_df_edges(G=G, df_nodes=df_sub_nodes, df_edges=df_edges, target_node=target_node)
        visualize_contextualized_supply_chain_subgraph(env=env, event_dict=event_dict, target_node=target_node, df_edges=df_simp_edges, df_nodes=df_sub_nodes, path=f"{save_path}/{env_config_name}/graph_imgs/{data_idx+n_cum_questions}.png")
        df_sub_nodes.to_csv(f"{save_path}/{env_config_name}/nodes/{data_idx+n_cum_questions}.csv", index=False)
        df_simp_edges.to_csv(f"{save_path}/{env_config_name}/edges/{data_idx+n_cum_questions}.csv", index=False)

    return questions, answers


def generate_price_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env: dict, data_idx: int, num_questions: int=10):
    
    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    t = env['t']
    questions = []
    answers = []
    n_cum_questions = 0

    while n_cum_questions < num_questions:
        
        target_node_stage_id = random.choice(range(num_stages-1))
        target_node_agent_id = random.choice(range(num_agents_per_stage))
        target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"
        question = f"Your are {target_node} at round {t}. Based on the provided supply chain graph, which of the upstream agents at stage {target_node_stage_id+1} offer the lowerest price? Answer the name of the , e.g. [stage_1_agent_1]."
        
        answer_agent_id = np.argmin(env['sale_prices'][(target_node_stage_id+1)*num_agents_per_stage:(target_node_stage_id+2)*num_agents_per_stage])
        answer = f"stage_{target_node_stage_id+1}_agent_{answer_agent_id}"

        # Save the target-node-related graph as node df/edge df/graph/graph img
        # df_sub_nodes.to_csv(f"{save_path}/{env_config_name}/nodes/{data_idx*num_questions+n_cum_questions}.csv", index=False)
        df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node, path=f"{save_path}/{env_config_name}/nodes/{data_idx+n_cum_questions}.csv")
        df_simp_edges = get_price_sub_df_edges(G=G, df_nodes=df_sub_nodes, df_edges=df_edges, target_node=target_node, path=f"{save_path}/{env_config_name}/edges/{data_idx+n_cum_questions}.csv")
        visualize_contextualized_supply_chain_subgraph(env=env, event_dict=event_dict, target_node=target_node, df_edges=df_simp_edges, df_nodes=df_sub_nodes, path=f"{save_path}/{env_config_name}/graph_imgs/{data_idx+n_cum_questions}.png")
        df_sub_nodes.to_csv(f"{save_path}/{env_config_name}/nodes/{data_idx+n_cum_questions}.csv", index=False)
        df_simp_edges.to_csv(f"{save_path}/{env_config_name}/edges/{data_idx+n_cum_questions}.csv", index=False)
        questions.append(question)
        answers.append(answer)
        n_cum_questions += 1

    return questions, answers


def generate_lead_time_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env: dict, data_idx: int, num_questions: int=10):
    
    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    t = env['t']
    questions = []
    answers = []
    n_cum_questions = 0

    while n_cum_questions < num_questions:
        try:
            target_node_stage_id = random.choice(range(num_stages-1))
            target_node_agent_id = random.choice(range(num_agents_per_stage))
            target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"
            question = f"Your are {target_node} at round {t}. Based on the provided supply chain graph, which of the upstream agents at stage {target_node_stage_id+1} has the shortest lead time? Answer the name of the , e.g. [stage_1_agent_1]."
            answer_agent_id = np.argmin(env['lead_times'][target_node_stage_id][target_node_agent_id])
            answer = f"stage_{target_node_stage_id+1}_agent_{answer_agent_id}"

            # Save the target-node-related graph as node df/edge df/graph/graph img
            df_nodes.to_csv("the one with problem.csv", index=False)
            df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node)
            df_simp_edges = get_lt_sub_df_edges(G=G, df_nodes=df_sub_nodes, df_edges=df_edges, target_node=target_node)
            visualize_contextualized_supply_chain_subgraph(env=env, event_dict=event_dict, target_node=target_node, df_edges=df_simp_edges, df_nodes=df_sub_nodes, path=f"{save_path}/{env_config_name}/graph_imgs/{data_idx+n_cum_questions}.png")
            df_sub_nodes.to_csv(f"{save_path}/{env_config_name}/nodes/{data_idx+n_cum_questions}.csv", index=False)
            df_simp_edges.to_csv(f"{save_path}/{env_config_name}/edges/{data_idx+n_cum_questions}.csv", index=False)
            questions.append(question)
            answers.append(answer)
            n_cum_questions += 1
        except:
            pass

    return questions, answers

# TODO: assign demand for each agents. Add more attr in df_nodes
def generate_demand_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env: dict, data_idx: int, num_questions: int=10):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    questions = []
    answers = []
    t = env['t']
    demand_t_1 = [env['demand_fn'](t+1) for _ in range(10)]
    num_init_suppliers = env['num_init_suppliers']
    supply_relations = env['supply_relations']
    mean_demand_t_1, std_demand_t_1 = np.mean(demand_t_1), np.std(demand_t_1)
    n_cum_questions = 0

    while n_cum_questions < num_questions:
        try:
            target_node_stage_id = random.choice(range(1, num_stages))
            target_node_agent_id = random.choice(range(num_agents_per_stage))
            target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"
            
            question = f"Your are {target_node} at round {t}. Based on the order quantity from the downstream customers in the supply chain graph, what is your estimated demand for the future rounds? Answer the number of units e.g., [10]"
            num_customers = sum(supply_relations[target_node_stage_id-1][:, target_node_agent_id])
            ratio = num_customers/num_init_suppliers
            answer = (np.floor((mean_demand_t_1-std_demand_t_1)*ratio), np.ceil((mean_demand_t_1+std_demand_t_1)*ratio))
    
            # Save the target-node-related graph as node df/edge df/graph/graph img
            df_edges.to_csv("the one with problem.csv", index=False)
            df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node, path=f"{save_path}/{env_config_name}/nodes/{data_idx+n_cum_questions}.csv")
            df_simp_edges = get_sub_df_edges(G=G, df_nodes=df_sub_nodes, df_edges=df_edges, target_node=target_node, path=f"{save_path}/{env_config_name}/edges/{data_idx+n_cum_questions}.csv")
            visualize_contextualized_supply_chain_subgraph(env=env, event_dict=event_dict, target_node=target_node, df_edges=df_simp_edges, df_nodes=df_sub_nodes, path=f"{save_path}/{env_config_name}/graph_imgs/{data_idx+n_cum_questions}.png")
            df_sub_nodes.to_csv(f"{save_path}/{env_config_name}/nodes/{data_idx+n_cum_questions}.csv", index=False)
            df_simp_edges.to_csv(f"{save_path}/{env_config_name}/edges/{data_idx+n_cum_questions}.csv", index=False)
            questions.append(question)
            answers.append(answer)
            n_cum_questions += 1
        except:
            pass

    return questions, answers


def generate_supplier_choice_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env: dict, data_idx: int, num_questions: int=10):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    questions = []
    answers = []
    t = env['t']
    n_cum_questions = 0

    while n_cum_questions < num_questions:        
        try:

            target_node_stage_id = random.choice(range(num_stages-1))
            target_node_agent_id = random.choice(range(num_agents_per_stage))
            target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"
            question = (
                            f"Your are {target_node} at round {t}. Based on the provided supply chain graph, which of the upstream companies at stage {target_node_stage_id+1} you would like to choose as your suppliers in the next round?"
                            "Please consider the price, lead time, order fullfillment in the previous rounds, the potential effect from the emergent events. Answer the name of the suppliers, e.g. (stage_x_agent_y)."
                        )
            answer = rank_suppliers_by_reliability(G=G, env=env, stage_idx=target_node_stage_id, agent_idx=target_node_agent_id)
            answer = dict(zip([f"stage_{target_node_stage_id+1}_agent_{i}" for i in range(num_agents_per_stage)], answer))
            # Save the target-node-related graph as node df/edge df/graph/graph img
            df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node)
            df_simp_edges = get_sub_df_edges(G=G, df_edges=df_edges, df_nodes=df_sub_nodes, target_node=target_node, path=f"{save_path}/{env_config_name}/edges/{data_idx+n_cum_questions}.csv")
            visualize_contextualized_supply_chain_subgraph(env=env, event_dict=event_dict, target_node=target_node, df_edges=df_simp_edges, df_nodes=df_sub_nodes, path=f"{save_path}/{env_config_name}/graph_imgs/{data_idx+n_cum_questions}.png")
            df_sub_nodes.to_csv(f"{save_path}/{env_config_name}/nodes/{data_idx+n_cum_questions}.csv", index=False)
            df_simp_edges.to_csv(f"{save_path}/{env_config_name}/edges/{data_idx+n_cum_questions}.csv", index=False)
            n_cum_questions += 1
            questions.append(question)
            answers.append(answer)

        except:
            pass


    return questions, answers
                                                                                                                                                                                                          

def rank_suppliers_by_reliability(G: nx.DiGraph, env: dict, stage_idx: int, agent_idx: int):

    supp_stage_idx = stage_idx + 1
    num_agents_per_stage = env['num_agents_per_stage']
    # num_stages = env['num_stages']
    sale_prices = env['sale_prices'][supp_stage_idx*num_agents_per_stage:(supp_stage_idx+1)*num_agents_per_stage]
    lead_times = env['lead_times'][stage_idx, agent_idx]
    order_fulfill_rates = env['order_fulfill_rates'][supp_stage_idx, :, agent_idx]
    
    # rank suppliers by price
    price_rank_score = rankdata(sale_prices, method='min')
    # rank suppliers by lead time
    lead_times_rank_score = rankdata(lead_times, method='min')
    # rank suppliers by order fulfillment
    order_fulfillments_rank_score = rankdata(-order_fulfill_rates, method='min')
    # rank suppliers by the event effect
    event_effect_rank = np.zeros(num_agents_per_stage)
    for event, affected_agents in env['events'].items():
        sid, aid = affected_agents
        for supp_agent_idx in range(num_agents_per_stage):
            if check_connection(G=G, event_target_node=f"stage_{sid}_agent_{aid}", target_node=f"stage_{supp_stage_idx}_agent_{supp_agent_idx}"):
                event_effect_rank[supp_agent_idx] += 1
    event_effect_rank_score = rankdata(event_effect_rank, method='min')

    return price_rank_score + lead_times_rank_score + order_fulfillments_rank_score + event_effect_rank_score


if __name__ == "__main__":

    
    events_list = read_data_from_json(read_path="src/gnn/gnn_dataset/supply_chain_events.json")
    event_dict = {"events": [x[0] for x in events_list[1:]],
                "Type": [x[1] for x in events_list[1:]],
                "Aspect": [x[2] for x in events_list[1:]]}
    num_events = len(event_dict['events'])


    env_config_name = "large_graph_test"
    save_path = f"src/gnn/gnn_dataset"
    create_event_questions = True
    create_price_questions = False
    create_lead_time_questions = False
    create_supplier_questions = True
    os.makedirs(f"{save_path}/{env_config_name}/nodes", exist_ok=True)
    clear_dir(f"{save_path}/{env_config_name}/nodes")
    os.makedirs(f"{save_path}/{env_config_name}/edges", exist_ok=True)
    clear_dir(f"{save_path}/{env_config_name}/edges")
    os.makedirs(f"{save_path}/{env_config_name}/graphs", exist_ok=True)
    clear_dir(f"{save_path}/{env_config_name}/graphs")
    os.makedirs(f"{save_path}/{env_config_name}/graph_imgs", exist_ok=True)
    clear_dir(f"{save_path}/{env_config_name}/graph_imgs")

    df_event_qa = pd.DataFrame({"question": [], "label": []})
    df_price_qa = pd.DataFrame({"question": [], "label": []})
    df_lead_time_qa = pd.DataFrame({"question": [], "label": []})
    df_suppliers_qa = pd.DataFrame({"question": [], "label": []})
    
    num_graphs = 200
    num_questions_per_graph = 5
    data_idx = 0
    for _ in tqdm(range(num_graphs)):
    # for data_idx in tqdm(range(num_graphs)):
        env = generate_env(env_config_name=env_config_name)
        num_stages = env['num_stages']
        num_agents_per_stage = env['num_agents_per_stage']

        events = assign_events(num_events, num_stages, num_agents_per_stage)
        env['events'] = events
        df_nodes = convert_env_to_node_df(env=env)
        df_edges = convert_env_to_edge_df(env=env, event_dict=event_dict)

        G = build_supplier_graph(df_edges=df_edges, df_nodes=df_nodes)
        
        if create_event_questions:
            questions, answers = generate_event_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, data_idx=data_idx)
            df_event_qa = pd.concat([df_event_qa, pd.DataFrame({"question": questions, 
                                                    "label": answers, 
                                                    'graph_idx': np.arange(data_idx, data_idx+num_questions_per_graph)})], axis=0)       
        data_idx += num_questions_per_graph

        if create_price_questions:
            questions, answers = generate_price_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, data_idx=data_idx)
            df_price_qa = pd.concat([df_price_qa, pd.DataFrame({"question": questions, 
                                                    "label": answers, 
                                                    'graph_idx': np.arange(data_idx, data_idx+num_questions_per_graph)})], axis=0)
        data_idx += num_questions_per_graph

        if create_lead_time_questions:
            questions, answers = generate_lead_time_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, data_idx=data_idx)
            df_lead_time_qa = pd.concat([df_lead_time_qa, pd.DataFrame({"question": questions, 
                                                    "label": answers, 
                                                    'graph_idx': np.arange(data_idx, data_idx+num_questions_per_graph)})], axis=0)
        data_idx += num_questions_per_graph
        if create_supplier_questions:
            questions, answers = generate_supplier_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, data_idx=data_idx)
            df_lead_time_qa = pd.concat([df_suppliers_qa, pd.DataFrame({"question": questions,
                                                    "label": answers,
                                                    'graph_idx': np.arange(data_idx, data_idx+num_questions_per_graph)})], axis=0)
        data_idx += num_questions_per_graph
        
        
    if create_event_questions:
        df_event_qa['graph_idx'] = df_event_qa['graph_idx'].astype(int)
        df_event_qa.to_csv(f"{save_path}/{env_config_name}/all_event_questions.csv", index=False)

    if create_price_questions:
        df_price_qa['graph_idx'] = df_price_qa['graph_idx'].astype(int)
        df_price_qa.to_csv(f"{save_path}/{env_config_name}/all_price_questions.csv", index=False)

    if create_lead_time_questions:
        df_lead_time_qa['graph_idx'] = df_lead_time_qa['graph_idx'].astype(int)
        df_lead_time_qa.to_csv(f"{save_path}/{env_config_name}/all_lead_time_questions.csv", index=False)
    
    if create_supplier_questions:
        df_lead_time_qa['graph_idx'] = df_lead_time_qa['graph_idx'].astype(int)
        df_lead_time_qa.to_csv(f"{save_path}/{env_config_name}/all_supplier_questions.csv", index=False)

    # generate test questions
    # df_est_demand_qa = pd.DataFrame({"question": [], "label": []})
    # df_supplier_reliability_qa = pd.DataFrame({"question": [], "label": []})

    # num_test_graph = 20
    # num_questions_per_graph = 5
    # for _ in tqdm(range(num_test_graph)):

    #     env = generate_env(env_config_name=env_config_name)
    #     num_stages = env['num_stages']
    #     num_agents_per_stage = env['num_agents_per_stage']

    #     events = assign_events(num_events, num_stages, num_agents_per_stage)
    #     env['events'] = events
    #     df_nodes = convert_env_to_node_df(env=env)
    #     df_edges = convert_env_to_edge_df(env=env, event_dict=event_dict)

    #     G = build_supplier_graph(df_edges=df_edges, df_nodes=df_nodes)
    #     questions, answers = generate_demand_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, data_idx=data_idx)
    #     df_est_demand_qa = pd.concat([df_est_demand_qa, pd.DataFrame({"question": questions, 
    #                                             "label": answers, 
    #                                             'graph_idx': np.arange(data_idx, data_idx+num_questions_per_graph)})], axis=0)
    #     data_idx += num_questions_per_graph

    #     questions, answers = generate_supplier_choice_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, data_idx=data_idx)
    #     df_supplier_reliability_qa = pd.concat([df_supplier_reliability_qa, 
    #                                             pd.DataFrame({"question": questions, 
    #                                                             "label": answers, 
    #                                                             'graph_idx': range(data_idx, (data_idx+num_questions_per_graph))})], axis=0)
    #     data_idx += num_questions_per_graph

    # df_est_demand_qa['graph_idx'] = df_est_demand_qa['graph_idx'].astype(int)
    # df_est_demand_qa.to_csv(f"{save_path}/{env_config_name}/test_demand_questions.csv", index=False)

    # df_supplier_reliability_qa['graph_idx'] = df_supplier_reliability_qa['graph_idx'].astype(int)
    # df_supplier_reliability_qa.to_csv(f"{save_path}/{env_config_name}/test_supplier_questions.csv", index=False)

    # combine all_**_questions.csv to one
    df_event_qa = pd.read_csv(f"{save_path}/{env_config_name}/all_event_questions.csv")
    # df_price_qa = pd.read_csv(f"{save_path}/{env_config_name}/all_price_questions.csv")
    # df_lead_time_qa = pd.read_csv(f"{save_path}/{env_config_name}/all_lead_time_questions.csv")
    df_suppliers_qa = pd.read_csv(f"{save_path}/{env_config_name}/all_supplier_questions.csv")
    df_all_questions = pd.concat([df_event_qa, df_price_qa, df_lead_time_qa], axis=0)
    df_all_questions.to_csv(f"{save_path}/{env_config_name}/all_train_questions.csv", index=False)

    # df_est_demand_qa = pd.read_csv(f"{save_path}/{env_config_name}/test_demand_questions.csv")
    # df_supplier_reliability_qa = pd.read_csv(f"{save_path}/{env_config_name}/test_supplier_questions.csv")
    # df_test_questions = pd.concat([df_est_demand_qa, df_supplier_reliability_qa], axis=0)
    # df_test_questions.to_csv(f"{save_path}/{env_config_name}/all_test_questions.csv", index=False)
