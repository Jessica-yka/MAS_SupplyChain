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
import json
sys.path.append('/data/yanjia/MAS_SupplyChain')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.config import env_configs_list
from src.model.utils.utils import clear_dir, split_demand, save_data_to_json, read_data_from_json
from src.model.data_simulation import generate_lead_time, generate_prod_capacity, generate_backlogs
from src.model.data_simulation import generate_cost_price, generate_sup_dem_relations
from src.model.data_simulation import generate_holding_costs, generate_backlog_costs, generate_init_inventories
from src.model.data_simulation import Demand_fn
from src.gnn.preprocess.utils.retrieval import get_event_sub_df_edges, get_sub_df_nodes
from src.gnn.preprocess.utils.utils import rank_suppliers_by_reliability
import matplotlib.pyplot as plt
import networkx as nx
import random
import csv
from tqdm import tqdm
import torch
from scipy.stats import rankdata
from concurrent.futures import ThreadPoolExecutor
from src.gnn.preprocess.utils.utils import save_graph_to_json, save_env_to_json
import random
import argparse
from src.gnn.preprocess.events import events
np.random.seed(2025)

parser = argparse.ArgumentParser()
parser.add_argument('--for_train', action='store_true', default=False)
# Define the list of events with simplified descriptions


save_data_to_json(data=events, save_path="src/gnn/gnn_dataset/supply_chain_events.json")

print("CSV file 'supply_chain_events.csv' created successfully!")

# %% [markdown]
# ## Environment generation

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


def convert_env_to_node_df(env: dict, event_dict: dict):
    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    stage_names = env['stage_names']
    num_current_events = len(env['events'])
    num_nodes = 1 + num_stages * num_agents_per_stage + num_current_events
    df_node = pd.DataFrame(index=range(num_nodes), columns=["node_id", "type"])
    df_node["node_id"] = np.arange(num_nodes).tolist()
    df_node["name"] = ["Customers"] + [f"stage_{m}_agent_{x}" for m in range(num_stages) for x in range(num_agents_per_stage)] + [event_dict['events'][eidx] for eidx in env['events'].keys()]
    df_node["type"] = ["customers"] + [stage_names[m] for m in range(num_stages) for x in range(num_agents_per_stage)] + ["event" for _ in range(num_current_events)]
    df_node['sale_price'] = [0] + env['sale_prices'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['prod_capacity'] = [0] + env['prod_capacities'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['prod_cost'] = [0] + env['prod_costs'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['holding_cost'] = [0] + env['holding_costs'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['backlog_cost'] = [0] + env['backlog_costs'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['inventory'] = [0] + env['inventories'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['backlog'] = [0] + env['backlog'].flatten().tolist() + [0 for _ in range(num_current_events)]
    df_node['upstream_backlog'] = [0] + [0 for _ in range(num_stages*num_agents_per_stage)] + [0 for _ in range(num_current_events)]
    df_node['stage_id'] = [-1] + [m for m in range(num_stages) for _ in range(num_agents_per_stage)] + [-1 for _ in env['events'].keys()]
    df_node['agent_id'] = [-1] + [x for _ in range(num_stages) for x in range(num_agents_per_stage)] + [-1 for _ in env['events'].keys()]
    df_node['running_status'] = [1] + env['running_agents'].flatten().tolist() + [1 for _ in range(num_current_events)]
    df_node = df_node[df_node['running_status'] == 1].reset_index(drop=True)

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
    t = env['t']
    # Randomly create backlog events between suppliers and customers
    for m in range(num_stages-1):
        for x in range(num_agents_per_stage):
            for i in range(num_agents_per_stage):
                if sup_rel[m][x][i] == 1:
                    num_request_order = env['demand_fn'](t)//num_init_suppliers
                    num_fufilled_order = int(num_request_order * order_fulfill_rates[m+1][i][x]) # the fulfillment rate of the supplier stage_(m+1)_agent_i to the customer stage_m_agent_x
                    is_fulfilled = "yes" if num_fufilled_order >= num_request_order else "no"
                    df_edge.loc[edge_idx, ["source", "target", "label", "type", 'aspect']] = \
                        [f"stage_{m}_agent_{x}", f"stage_{m+1}_agent_{i}", f"request order of {num_request_order} units of product at round {t-1}", "", []]
                    env['requested_order'][m+1][i] += num_request_order
                    df_edge.loc[edge_idx+1, ["source", "target", "label", 'type', 'aspect']] = \
                        [f"stage_{m+1}_agent_{i}", f"stage_{m}_agent_{x}", f"deliverying {num_fufilled_order} units of product at round {t}", is_fulfilled, []]
                    edge_idx += 2

    # Create downstream demand for retailers
    for x in range(num_agents_per_stage):
        num_request_order = env['demand_fn'](t)
        df_edge.loc[edge_idx, ["source", "target", "label", "type", 'aspect']] = \
            [f"Customers", f"stage_0_agent_{x}", f"have a demand for {num_request_order} units of product at round {t-1}", "", []]
        env['requested_order'][0][x] += num_request_order
        edge_idx += 1

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
    
    # add lead time info to the edge_df
    for stage_id in range(num_stages-1):
        for agent_id in range(num_agents_per_stage):
            for target_agent_id in range(num_agents_per_stage):
                lt = env['lead_times'][stage_id][agent_id][target_agent_id]
                df_edge.loc[edge_idx, ["source", "target", "label", "type", "aspect"]] = \
                        [f"stage_{stage_id+1}_agent_{target_agent_id}", f"stage_{stage_id}_agent_{agent_id}", f"has lead time of {lt} days to", "", []]
                edge_idx += 1

    # add potential suppliers info to the edge_df
    for stage_id in range(num_stages-1):
        for agent_id in range(num_agents_per_stage):
            for target_agent_id in range(num_agents_per_stage):
                df_edge.loc[edge_idx, ["source", "target", "label", "type", "aspect"]] = \
                        [f"stage_{stage_id+1}_agent_{target_agent_id}", f"stage_{stage_id}_agent_{agent_id}", f"is an upstream agent to", "", []]
                edge_idx += 1

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
    running_agents = env_configs

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
            't': random.choice(range(1, num_periods-1)),
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
            'requested_order': np.zeros((num_stages, num_agents_per_stage)), # because each agent has only one supplier, so l reduce the 3d array to 2d array
            'running_agents': np.ones((num_stages, num_agents_per_stage)), # assume all agents are working
        }

# %%
## Randomly generate questions
# 1. assign the target node
# 2. Question: Whether the event would affect the target node positively or negatively in terms of the aspect
# 3. Search the graph to see if there is a path from the event to the target node (omit the direction, just check the connection)
# 4. If there is a path, the answer is  positive/negative (based on the aspect)
# 5. If there is no path, the answer is "neutral"
# 6. Generate 10 questions
def generate_target_node(num_stages: int, num_agents_per_stage: int):

    target_node_stage_id = random.choice(range(num_stages))
    target_node_agent_id = random.choice(range(num_agents_per_stage))

    return target_node_stage_id, target_node_agent_id

def list_all_successor_nodes(G, node):
    # select successor based on "is supplier of" relation only
    return [n for n in nx.nodes(nx.dfs_tree(G, node)) if n.split("_")[1] < node.split("_")[1]]

def list_all_predecessor_nodes(G, node):
    return [n for n in nx.nodes(nx.dfs_tree(G.reverse(), node)) if 'stage' in n and n.split("_")[1] > node.split("_")[1]]

def generate_event_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env: dict, data_idx:int, num_questions:int=5):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    t = env['t']
    # aspect_list = ["Production Capacity", "Delivery Time", "Order Fulfillment", "Price", "Demand"]
    # up_aspect_list = ["Production Capacity", "Delivery Time", "Order Fulfillment", "Price"]
    # down_aspect_list = ["Demand"]

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
    cur_target_nodes = []

    n_cum_questions = 0
    while n_cum_questions < num_questions:
        try:
            # Get a valid event
            es = event_aspect_in_graph[n_cum_questions%num_event_aspect_in_graph]
            df_event_aspect = df_edges[df_edges['aspect'].apply(lambda x: es in x)].reset_index(drop=True)
            row_id = random.choice(range(len(df_event_aspect)))
            # event_aspect = df_event_aspect.loc[row_id, 'aspect']

            event_type = df_event_aspect.loc[row_id, 'type']
            event_target_node = df_event_aspect.loc[row_id, 'target']
            event_node = df_event_aspect.loc[row_id, 'source']

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
                
                event_target_node_relation = random.choice(['upstream suppliers', 'downstream customers'])
                # asp = random.choice(down_aspect_list) if node_rel == 'customers' else random.choice(up_aspect_list)
                question = f"You are {target_node} at round {t}. Based on the provided supply chain graph, how would the {event_node} affect your {event_target_node_relation}? Answer either 'positive' or 'negative' if it happens to your {event_target_node_relation}(s), otherwise answer 'neutral'."
                answer = "neutral"

            elif cases == 'downstream':
                # asp = random.choice(down_aspect_list)
                try:
                    target_node = random.choice(list_all_predecessor_nodes(G=G, node=event_target_node))
                except:
                    continue
                question = f"You are {target_node} at round {t}. Based on the provided supply chain graph, how would the {event_node} affect your downstream customers? Answer either 'positive' or 'negative' if it happens to your customer(s), otherwise answer 'neutral'."
                answer = "positive" if event_type == "Positive" else "negative"
            else:
                # asp = random.choice(up_aspect_list)
                try:
                    target_node = random.choice(list_all_successor_nodes(G=G, node=event_target_node))
                except:
                    continue
                question = f"Your are {target_node} at round {t}. Based on the provided supply chain graph, how would the {event_node} affect your upstream suppliers? Answer either 'positive' or 'negative' if it happens to your supplier(s), otherwise answer 'neutral'."
                answer = "positive" if event_type == "Positive" else "negative"
            
            # Save the target-node-related graph as node df/edge df/graph/graph img
            df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node)
            df_simp_edges = get_event_sub_df_edges(G=G, df_nodes=df_sub_nodes, df_edges=df_edges, target_node=target_node)
            # visualize_contextualized_supply_chain_subgraph(env=env, target_node=target_node, df_edges=df_simp_edges, df_nodes=df_sub_nodes, path=f"{save_graph_img_path}/{data_idx+n_cum_questions}.png")
            # df_sub_nodes.to_csv(f"{save_node_path}/{data_idx+n_cum_questions}.csv", index=False)
            # df_simp_edges.to_csv(f"{save_edge_path}/{data_idx+n_cum_questions}.csv", index=False)

            df_nodes.to_csv(f"{save_node_path}/{data_idx+n_cum_questions}.csv", index=False)
            df_edges.to_csv(f"{save_edge_path}/{data_idx+n_cum_questions}.csv", index=False)
            save_env_to_json(env, f"{save_env_path}/{data_idx+n_cum_questions}.json")
            save_graph_to_json(G, f"{save_G_path}/{data_idx+n_cum_questions}.json")
        except:
            pass

        questions.append(question)
        answers.append(answer)
        cur_target_nodes.append(target_node)
        n_cum_questions += 1


    return questions, answers, cur_target_nodes


def generate_price_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env: dict, data_idx: int, target_nodes: list=None, num_questions: int=10):
    
    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    t = env['t']
    questions = []
    answers = []
    cur_target_nodes = []
    n_cum_questions = 0

    while n_cum_questions < num_questions:
        if target_nodes is None:
            target_node_stage_id = random.choice(range(num_stages-1))
            target_node_agent_id = random.choice(range(num_agents_per_stage))
            target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"
        else:
            target_node = target_nodes[n_cum_questions]
            _, target_node_stage_id, _, target_node_agent_id = target_node.split("_")
            target_node_stage_id = int(target_node_stage_id)
            target_node_agent_id = int(target_node_agent_id)

        cur_target_nodes.append(target_node)
        question = f"You are {target_node} at round {t}. Based on the provided supply chain graph, which of the upstream agents at stage {target_node_stage_id+1} offer the lowerest price? Answer with the node id, e.g. 15."
        
        if target_node_stage_id == num_stages-1: # in test set, it is possible that the target node is the last stage
            answer_agent_id = 0
            # answer_agent_id = np.argmin(env['sale_prices'][(target_node_stage_id+1)*num_agents_per_stage:])
        else:
            answer_agent_id = np.argmin(env['sale_prices'][(target_node_stage_id+1)*num_agents_per_stage:(target_node_stage_id+2)*num_agents_per_stage])

        answer = str(node_name_id_map[f"stage_{target_node_stage_id+1}_agent_{answer_agent_id}"])

        # Save the target-node-related graph as node df/edge df/graph/graph img
        # df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node)
        # df_simp_edges = get_price_sub_df_edges(df_nodes=df_sub_nodes, df_edges=df_edges, target_node=target_node)
        # visualize_contextualized_supply_chain_subgraph(env=env, target_node=target_node, df_edges=df_simp_edges, df_nodes=df_sub_nodes, path=f"{save_graph_img_path}/{data_idx+n_cum_questions}.png")
        # df_sub_nodes.to_csv(f"{save_node_path}/{data_idx+n_cum_questions}.csv", index=False)
        # df_simp_edges.to_csv(f"{save_edge_path}/{data_idx+n_cum_questions}.csv", index=False)

        df_nodes.to_csv(f"{save_node_path}/{data_idx+n_cum_questions}.csv", index=False)
        df_edges.to_csv(f"{save_edge_path}/{data_idx+n_cum_questions}.csv", index=False)
        save_env_to_json(env, f"{save_env_path}/{data_idx+n_cum_questions}.json")
        save_graph_to_json(G, f"{save_G_path}/{data_idx+n_cum_questions}.json")
        questions.append(question)
        answers.append(answer)
        n_cum_questions += 1

    return questions, answers, cur_target_nodes


def generate_lead_time_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env: dict, data_idx: int, target_nodes: list=None, num_questions: int=10):
    
    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    t = env['t']
    questions = []
    answers = []
    n_cum_questions = 0
    cur_target_nodes = []

    while n_cum_questions < num_questions:
        if target_nodes is None:
            target_node_stage_id = random.choice(range(num_stages-1))
            target_node_agent_id = random.choice(range(num_agents_per_stage))
            target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"
        else:
            target_node = target_nodes[n_cum_questions]
            _, target_node_stage_id, _, target_node_agent_id = target_node.split("_")
            target_node_stage_id = int(target_node_stage_id)
            target_node_agent_id = int(target_node_agent_id)

        
        question = f"You are {target_node} at round {t}. Based on the provided supply chain graph, which of the upstream agents at stage {target_node_stage_id+1} has the shortest lead time? Answer with the node id, e.g. 17."
        if target_node_stage_id == num_stages-1: # in test set, it is possible that the target node is the last stage
            answer_agent_id = 0
        else:
            answer_agent_id = np.argmin(env['lead_times'][target_node_stage_id][target_node_agent_id])

        answer = node_name_id_map[f"stage_{target_node_stage_id+1}_agent_{answer_agent_id}"]

        # Save the target-node-related graph as node df/edge df/graph/graph img
        # df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node)
        # df_simp_edges = get_lt_sub_df_edges(df_nodes=df_sub_nodes, df_edges=df_edges, target_node=target_node)
        # visualize_contextualized_supply_chain_subgraph(env=env, target_node=target_node, df_edges=df_simp_edges, df_nodes=df_sub_nodes, path=f"{save_graph_img_path}/{data_idx+n_cum_questions}.png")
        # df_sub_nodes.to_csv(f"{save_node_path}/{data_idx+n_cum_questions}.csv", index=False)
        # df_simp_edges.to_csv(f"{save_edge_path}/{data_idx+n_cum_questions}.csv", index=False)

        df_nodes.to_csv(f"{save_node_path}/{data_idx+n_cum_questions}.csv", index=False)
        df_edges.to_csv(f"{save_edge_path}/{data_idx+n_cum_questions}.csv", index=False)
        save_env_to_json(env, f"{save_env_path}/{data_idx+n_cum_questions}.json")
        save_graph_to_json(G, f"{save_G_path}/{data_idx+n_cum_questions}.json")
        
        questions.append(question)
        answers.append(answer)
        cur_target_nodes.append(target_node)
        n_cum_questions += 1

    return questions, answers, cur_target_nodes

def generate_orderFulfill_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env:dict, data_idx:int, target_nodes: list=None, num_questions: int=5):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    t = env['t']
    questions = []
    answers = []
    n_cum_questions = 0
    cur_target_nodes = []

    while n_cum_questions < num_questions:
        if target_nodes is None:
            target_node_stage_id = random.choice(range(num_stages-1))
            target_node_agent_id = random.choice(range(num_agents_per_stage))
            target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"
        else:
            target_node = target_nodes[n_cum_questions]
            _, target_node_stage_id, _, target_node_agent_id = target_node.split("_")
            target_node_stage_id = int(target_node_stage_id)
            target_node_agent_id = int(target_node_agent_id)

        
        supp_node_agent_id = np.argmax(env['supply_relations'][target_node_stage_id][target_node_agent_id])
        question = f"You are {target_node} at round {t}. Considering the provided supply chain graph, is your supplier stage_{target_node_stage_id+1}_agent_{supp_node_agent_id} meeting the order fulfillment by delivering the full requested amount of products in your order at round {t-1}? Please answer with either 'yes' or 'no'."
        if target_node_stage_id < num_stages - 1:
            answer = "yes" if env['order_fulfill_rates'][target_node_stage_id+1][supp_node_agent_id][target_node_agent_id] == 1 else "no"
        else:
            answer = "None"

        # Save the target-node-related graph as node df/edge df/graph/graph img
        # df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node)
        # df_simp_edges = get_of_sub_df_edges(df_nodes=df_sub_nodes, df_edges=df_edges, target_node=target_node)
        # visualize_contextualized_supply_chain_subgraph(env=env, target_node=target_node, df_edges=df_simp_edges, df_nodes=df_sub_nodes, path=f"{save_graph_img_path}/{data_idx+n_cum_questions}.png")
        # df_sub_nodes.to_csv(f"{save_node_path}/{data_idx+n_cum_questions}.csv", index=False)
        # df_simp_edges.to_csv(f"{save_edge_path}/{data_idx+n_cum_questions}.csv", index=False)

        df_nodes.to_csv(f"{save_node_path}/{data_idx+n_cum_questions}.csv", index=False)
        df_edges.to_csv(f"{save_edge_path}/{data_idx+n_cum_questions}.csv", index=False)
        save_env_to_json(env, f"{save_env_path}/{data_idx+n_cum_questions}.json")
        save_graph_to_json(G, f"{save_G_path}/{data_idx+n_cum_questions}.json")
        
        questions.append(question)
        answers.append(answer)
        cur_target_nodes.append(target_node)
        n_cum_questions += 1

    return questions, answers, cur_target_nodes


def generate_demand_questions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env:dict, data_idx: int, target_nodes: list=None, num_questions: int=5):

    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    t = env['t']
    questions = []
    answers = []
    cur_target_nodes = []
    n_cum_questions = 0

    while n_cum_questions < num_questions:
        if target_nodes is None:
            target_node_stage_id = random.choice(range(num_stages))
            target_node_agent_id = random.choice(range(num_agents_per_stage))
            target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"
        else:
            target_node = target_nodes[n_cum_questions]
            _, target_node_stage_id, _, target_node_agent_id = target_node.split("_")
            target_node_stage_id = int(target_node_stage_id)
            target_node_agent_id = int(target_node_agent_id)

        
        if target_node_stage_id == 0:
            question = f"You are {target_node} at round {t-1}. Based on the provided supply chain graph, do you have sufficient inventory to fulfill the downstream demand at round {t-1}? Answer with either 'yes' or 'no'."
        else:
            question = f"You are {target_node} at round {t-1}. Based on the provided supply chain graph, do you have sufficient inventory to fulfill the requested order at round {t-1}? Answer with either 'yes' or 'no'."
        if env['requested_order'][target_node_stage_id][target_node_agent_id] <= env['inventories'][target_node_stage_id*num_agents_per_stage+target_node_agent_id]:
            answer = "yes"
        else:
            answer = 'no'

        df_nodes.to_csv(f"{save_node_path}/{data_idx+n_cum_questions}.csv", index=False)
        df_edges.to_csv(f"{save_edge_path}/{data_idx+n_cum_questions}.csv", index=False)
        save_env_to_json(env, f"{save_env_path}/{data_idx+n_cum_questions}.json")
        save_graph_to_json(G, f"{save_G_path}/{data_idx+n_cum_questions}.json")
        
        questions.append(question)
        answers.append(answer)
        cur_target_nodes.append(target_node)
        n_cum_questions += 1

    return questions, answers, cur_target_nodes


if __name__ == "__main__":

    
    events_list = read_data_from_json(read_path="src/gnn/gnn_dataset/supply_chain_events.json")
    event_dict = {"events": [x[0] for x in events_list[1:]],
                "Type": [x[1] for x in events_list[1:]],
                "Aspect": [x[2] for x in events_list[1:]]}
    num_events = len(event_dict['events'])

    args = parser.parse_args()
    for_train = args.for_train

    data_type = "train" if for_train else "test"
    env_config_name = "graph_4_4"
    save_path = f"src/gnn/gnn_dataset"
    create_event_questions = True
    create_price_questions = True
    create_lead_time_questions = True
    create_order_fulfill_questions = True
    create_demand_questions = True
    
    save_node_path = f"{save_path}/{env_config_name}/{data_type}_data/original_nodes"
    save_edge_path = f"{save_path}/{env_config_name}/{data_type}_data/original_edges"
    save_graph_path = f"{save_path}/{env_config_name}/{data_type}_data/graphs"
    save_graph_img_path = f"{save_path}/{env_config_name}/{data_type}_data/graph_imgs"
    save_env_path = f"{save_path}/{env_config_name}/{data_type}_data/envs"
    save_G_path = f"{save_path}/{env_config_name}/{data_type}_data/G"
    os.makedirs(save_node_path, exist_ok=True)
    clear_dir(save_node_path)
    os.makedirs(save_edge_path, exist_ok=True)
    clear_dir(save_edge_path)
    os.makedirs(save_graph_path, exist_ok=True)
    clear_dir(save_graph_path)
    os.makedirs(save_graph_img_path, exist_ok=True)
    clear_dir(save_graph_img_path)
    os.makedirs(save_env_path, exist_ok=True)
    clear_dir(save_env_path)
    os.makedirs(save_G_path, exist_ok=True)
    clear_dir(save_G_path)

    df_event_qa = pd.DataFrame({"question": [], "label": []})
    df_price_qa = pd.DataFrame({"question": [], "label": []})
    df_lead_time_qa = pd.DataFrame({"question": [], "label": []})
    df_order_fulfill_qa = pd.DataFrame({"question": [], "label": []})
    df_demand_qa = pd.DataFrame({"question": [], "label": []})
    df_suppliers_rank = {}
    
    
    if for_train:
        num_graphs = 200
        num_questions_per_graph = 5
    else:
        num_graphs = 100
        num_questions_per_graph = 1
    graph_idx = 0
    data_idx = 0
    progress_bar_test = tqdm(range(num_graphs))
    while graph_idx < num_graphs:

        env = generate_env(env_config_name=env_config_name)
        num_stages = env['num_stages']
        num_agents_per_stage = env['num_agents_per_stage']

        events = assign_events(num_events, num_stages, num_agents_per_stage)
        env['events'] = events
        df_nodes = convert_env_to_node_df(env=env, event_dict=event_dict)
        node_name_id_map = dict(zip(df_nodes['name'].tolist(), df_nodes['node_id'].tolist()))
        df_edges = convert_env_to_edge_df(env=env, event_dict=event_dict)

        G = build_supplier_graph(df_edges=df_edges, df_nodes=df_nodes)
        target_nodes = None
        if create_event_questions:
            # print("generate event questions")
            questions, answers, target_nodes = generate_event_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, data_idx=data_idx)
            df_event_qa = pd.concat([df_event_qa, pd.DataFrame({"question": questions, 
                                                                "label": answers, 
                                                                'graph_idx': np.arange(data_idx, data_idx+num_questions_per_graph),
                                                                "target_node": target_nodes})], axis=0)
            
        data_idx += num_questions_per_graph

        if create_price_questions:
            # print("generate price questions")
            if for_train:
                questions, answers, target_nodes = generate_price_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, target_nodes=None, data_idx=data_idx)
            else:
                questions, answers, _ = generate_price_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, target_nodes=target_nodes, data_idx=data_idx)
            df_price_qa = pd.concat([df_price_qa, pd.DataFrame({"question": questions, 
                                                                "label": answers, 
                                                                'graph_idx': np.arange(data_idx, data_idx+num_questions_per_graph),
                                                                "target_node": target_nodes})], axis=0)
        data_idx += num_questions_per_graph

        if create_lead_time_questions:
            # print("generate lead time questions")
            if for_train:
                questions, answers, target_nodes = generate_lead_time_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, target_nodes=None, data_idx=data_idx)
            else:
                questions, answers, _ = generate_lead_time_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, target_nodes=target_nodes, data_idx=data_idx)
            df_lead_time_qa = pd.concat([df_lead_time_qa, pd.DataFrame({"question": questions, 
                                                                        "label": answers, 
                                                                        'graph_idx': np.arange(data_idx, data_idx+num_questions_per_graph),
                                                                        "target_node": target_nodes})], axis=0)
        data_idx += num_questions_per_graph

        if create_order_fulfill_questions:
            # print("generate supplier questions")
            if for_train:
                questions, answers, target_nodes = generate_orderFulfill_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, target_nodes=None, data_idx=data_idx)
            else:
                questions, answers, _ = generate_orderFulfill_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, target_nodes=target_nodes, data_idx=data_idx)
            df_order_fulfill_qa = pd.concat([df_order_fulfill_qa, pd.DataFrame({"question": questions,
                                                                                "label": answers,
                                                                                'graph_idx': np.arange(data_idx, data_idx+num_questions_per_graph),
                                                                                "target_node": target_nodes})], axis=0)
        data_idx += num_questions_per_graph

        if create_demand_questions:
            # print("generate demand questions")
            if for_train:
                questions, answers, target_nodes = generate_demand_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, target_nodes=None, data_idx=data_idx)
            else:
                questions, answers, _ = generate_demand_questions(df_nodes=df_nodes, df_edges=df_edges, env=env, num_questions=num_questions_per_graph, target_nodes=target_nodes, data_idx=data_idx)
            df_demand_qa = pd.concat([df_demand_qa, pd.DataFrame({"question": questions,
                                                                    "label": answers,
                                                                    'graph_idx': np.arange(data_idx, data_idx+num_questions_per_graph),
                                                                    "target_node": target_nodes})], axis=0)
        data_idx += num_questions_per_graph

        if not for_train:
            supplier_ranks = rank_suppliers_by_reliability(G=G, env=env, target_nodes=target_nodes,
                                                        use_event_rank=create_event_questions, use_price_rank=create_price_questions, use_lead_time_rank=create_lead_time_questions)
            for i in range(num_questions_per_graph):
                df_suppliers_rank[graph_idx+i] = supplier_ranks[i]

        graph_idx += 1
        progress_bar_test.update(1)
    progress_bar_test.close()

    if create_event_questions:
        df_event_qa['graph_idx'] = df_event_qa['graph_idx'].astype(int)
        df_event_qa.to_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_event_questions.csv", index=False)

    if create_price_questions:
        df_price_qa['graph_idx'] = df_price_qa['graph_idx'].astype(int)
        df_price_qa['label'] = df_price_qa['label'].astype(int)
        df_price_qa['label'] = df_price_qa['label'].astype(str)
        df_price_qa.to_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_price_questions.csv", index=False)

    if create_lead_time_questions:
        df_lead_time_qa['graph_idx'] = df_lead_time_qa['graph_idx'].astype(int)
        df_lead_time_qa['label'] = df_lead_time_qa['label'].astype(int)
        df_lead_time_qa['label'] = df_lead_time_qa['label'].astype(str)
        df_lead_time_qa.to_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_lead_time_questions.csv", index=False)
    
    if create_order_fulfill_questions:
        df_order_fulfill_qa['graph_idx'] = df_order_fulfill_qa['graph_idx'].astype(int)
        df_order_fulfill_qa.to_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_order_fulfill_questions.csv", index=False)

    if create_demand_questions:
        df_demand_qa['graph_idx'] = df_demand_qa['graph_idx'].astype(int)
        df_demand_qa.to_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_demand_questions.csv", index=False)


    # combine all_**_questions.csv to one
    df_event_qa = pd.read_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_event_questions.csv")
    df_price_qa = pd.read_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_price_questions.csv")
    df_lead_time_qa = pd.read_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_lead_time_questions.csv")
    df_order_fulfill_qa = pd.read_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_order_fulfill_questions.csv")
    df_demand_qa = pd.read_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_demand_questions.csv")
    df_all_questions = pd.concat([df_event_qa, df_price_qa, df_lead_time_qa, df_order_fulfill_qa, df_demand_qa], axis=0)
    df_all_questions.to_csv(f"{save_path}/{env_config_name}/{data_type}_data/all_questions.csv", index=False)
    if not for_train:
        print("save supplier ranking")
        with open(f"{save_path}/{env_config_name}/{data_type}_data/supplier_ranks.json", 'w') as f:
            json.dump(df_suppliers_rank, f, indent=4)

    print("Done")
