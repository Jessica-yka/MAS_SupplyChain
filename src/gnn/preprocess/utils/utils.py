
import os
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
import json
import pandas as pd

def check_connection(G, event_target_node, target_node):

    return nx.has_path(G, source=event_target_node, target=target_node)


def rank_suppliers_by_reliability(G: nx.DiGraph, env: dict, target_nodes: list, use_event_rank: bool, use_price_rank: bool, use_lead_time_rank: bool): 

    ranks = []
    num_agents_per_stage = env['num_agents_per_stage']
    for target_node in target_nodes:
        _, target_node_stage_id, _, target_node_agent_id = target_node.split('_')
        target_node_stage_id = int(target_node_stage_id)
        target_node_agent_id = int(target_node_agent_id)

        if target_node_stage_id == env['num_stages'] - 1:
            rank = np.zeros(num_agents_per_stage).tolist()
            ranks.append(rank)
        else:
            supp_stage_idx = target_node_stage_id + 1
            
            # num_stages = env['num_stages']
            sale_prices = env['sale_prices'][supp_stage_idx*num_agents_per_stage:(supp_stage_idx+1)*num_agents_per_stage]
            lead_times = env['lead_times'][target_node_stage_id, target_node_agent_id]
            order_fulfill_rates = env['order_fulfill_rates'][supp_stage_idx, :, target_node_agent_id]
            
            # rank suppliers by price
            if use_price_rank:
                price_rank_score = rankdata(sale_prices, method='min')
            else:
                price_rank_score = np.zeros(num_agents_per_stage)

            # rank suppliers by lead time
            if use_lead_time_rank:
                lead_times_rank_score = rankdata(lead_times, method='min')
            else:
                lead_times_rank_score = np.zeros(num_agents_per_stage)

            # rank suppliers by order fulfillment
            order_fulfillments_rank_score = np.zeros(num_agents_per_stage)
            order_fulfillments_rank_score = rankdata(-order_fulfill_rates, method='min')

            # rank suppliers by the event effect
            if use_event_rank:
                event_effect_rank = np.zeros(num_agents_per_stage)
                for event, affected_agents in env['events'].items():
                    sid, aid = affected_agents
                    for supp_agent_idx in range(num_agents_per_stage):
                        if check_connection(G=G, event_target_node=f"stage_{sid}_agent_{aid}", target_node=f"stage_{supp_stage_idx}_agent_{supp_agent_idx}"):
                            event_effect_rank[supp_agent_idx] += 1
                event_effect_rank_score = rankdata(event_effect_rank, method='min')
            else:
                event_effect_rank_score = np.zeros(num_agents_per_stage)

            rank = price_rank_score + lead_times_rank_score + order_fulfillments_rank_score + event_effect_rank_score
            ranks.append(rank.tolist())

    return ranks


# save nx.DiGraph to json
def save_graph_to_json(G: nx.DiGraph, path: str):
    data = nx.node_link_data(G)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
def load_json_to_graph(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)

def save_env_to_json(env: dict, path: str):
    sub_env = {}
    sub_env['num_stages'] = env['num_stages']
    sub_env['stage_names'] = env['stage_names']
    sub_env['num_agents_per_stage'] = env['num_agents_per_stage']
    sub_env['sale_prices'] = env['sale_prices'].tolist()
    sub_env['lead_times'] = env['lead_times'].tolist()
    sub_env['order_fulfill_rates'] = env['order_fulfill_rates'].tolist()
    sub_env['events'] = env['events']

    with open(path, 'w') as f:
        json.dump(sub_env, f, indent=4)

def load_json_to_env(path: str):
    with open(path, 'r') as f:
        env = json.load(f)
    return env