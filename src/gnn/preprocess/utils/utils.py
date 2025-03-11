
import os
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from .retrieval import check_connection
import json
import pandas as pd

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


def rank_suppliers_by_reliability(G: nx.DiGraph, env: dict, target_nodes: list, use_event_rank: bool, use_price_rank: bool, use_lead_time_rank: bool): 

    ranks = []
    for target_node in target_nodes:
        _, target_node_stage_id, _, target_node_agent_id = target_node.split('_')
        target_node_stage_id = int(target_node_stage_id)
        target_node_agent_id = int(target_node_agent_id)
        supp_stage_idx = target_node_stage_id + 1
        num_agents_per_stage = env['num_agents_per_stage']
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

        # print("price", price_rank_score.shape)
        # print("lead time", lead_times_rank_score.shape)
        # print("order", order_fulfillments_rank_score.shape)
        # print("event", event_effect_rank_score.shape)
        ranks.append(price_rank_score + lead_times_rank_score + order_fulfillments_rank_score + event_effect_rank_score)
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