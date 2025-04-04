import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import json
import dgl
from typing import Callable
import pickle



def save_chat_history_to_file(data: str, save_path: str, t: int, round: int=0):
    print("Saving data to: ", f"results/{save_path}/chat_results/chat_summary_round{round}_period{t}.txt")
    with open(f"results/{save_path}/chat_results/chat_summary_round{round}_period{t}.txt", 'w') as f:
        f.write(data)

def save_dict_to_json(data: dict, save_path: str):
    print("Saving config to: ", save_path)
    with open(save_path, 'w') as f:
        json.dump(data, f)

def save_data_to_json(data, save_path: str):
    print("Saving data to: ", save_path)
    with open(save_path, 'w') as f:
        json.dump(data, f)

def read_data_from_json(read_path: str):
    print("Reading data from: ", read_path)
    with open(read_path) as f:
        data = json.load(f)
    return data


def clear_dir(dir_path: str):
    # Clear the directory
    for file in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, file))

def save_array(data: np.ndarray, save_path: str):
    print("Saving data to: ", save_path)
    np.save(save_path, data)

def extract_pairs(input_string):
    # Remove unwanted characters and split the string
    """
    Extracts pairs in the format ("agentX", N) or ("agentX": N) from a string.
    
    Args:
        input_string (str): The input string containing pairs.
    
    Returns:
        list: A list of tuples containing the extracted pairs.
    """
    # Regular expression to match pairs like ("agentX", N) or ("agentX": N)
    pattern = r'\("([^"]+)"\s*[:,]\s*([0-9]+)\)'
    
    # Find all matches using the regex
    matches = re.findall(pattern, input_string)
    
    # Convert matches to tuples with proper format
    pairs = {agent: int(value) for agent, value in matches}
    
    return pairs

def parse_stage_agent_id(stage_agent_id_name: str):
    # Extract stage and agent from the string
    id_name = stage_agent_id_name.replace("agent_", "").replace("stage_", "")
    stage, agent = id_name.split("_")

    return int(stage), int(agent)


def create_action_dicts(env_config: str):
    print("Create action dictionaries for order, supply, demand, and price")
    # Create action dictionaries for order, supply, demand, and price
    all_action_order_dicts = {}
    all_action_sup_dicts = {}
    all_action_dem_dicts = {}
    all_action_price_dicts = {}
    all_reward_dicts = {}
    all_state_dicts = {}

    save_data_to_json(all_action_order_dicts, f"env/{env_config}/all_action_order_dicts.json")
    save_data_to_json(all_action_sup_dicts, f"env/{env_config}/all_action_sup_dicts.json")
    save_data_to_json(all_action_dem_dicts, f"env/{env_config}/all_action_dem_dicts.json")
    save_data_to_json(all_action_price_dicts, f"env/{env_config}/all_action_price_dicts.json")
    save_data_to_json(all_reward_dicts, f"env/{env_config}/all_reward_dicts.json")
    save_data_to_json(all_state_dicts, f"env/{env_config}/all_state_dicts.json")

    return 


def load_all_action_dicts(env_config: str):

    if os.path.exists(f"env/{env_config}/all_action_order_dicts.json"):
        all_action_order_dicts = read_data_from_json(f"env/{env_config}/all_action_order_dicts.json")
    else:
        all_action_order_dicts = {}
    if os.path.exists(f"env/{env_config}/all_action_sup_dicts.json"):
        all_action_sup_dicts = read_data_from_json(f"env/{env_config}/all_action_sup_dicts.json")
    else:
        all_action_sup_dicts = {}
    if os.path.exists(f"env/{env_config}/all_action_dem_dicts.json"):
        all_action_dem_dicts = read_data_from_json(f"env/{env_config}/all_action_dem_dicts.json")
    else:
        all_action_dem_dicts = {}
    if os.path.exists(f"env/{env_config}/all_action_price_dicts.json"):
        all_action_price_dicts = read_data_from_json(f"env/{env_config}/all_action_price_dicts.json")
    else:
        all_action_price_dicts = {}
    if os.path.exists(f"env/{env_config}/all_reward_dicts.json"):
        all_reward_dicts = read_data_from_json(f"env/{env_config}/all_reward_dicts.json")
    else:
        all_reward_dicts = {}
    if os.path.exists(f"env/{env_config}/all_state_dicts.json"):
        all_state_dicts = read_data_from_json(f"env/{env_config}/all_state_dicts.json")
    else:
        all_state_dicts = {}

    return all_action_order_dicts, all_action_sup_dicts, all_action_dem_dicts, all_action_price_dicts, all_reward_dicts, all_state_dicts

def load_action_dicts(env_config: str):

    action_sup_dicts = read_data_from_json(f"env/{env_config}/action_sup_dicts.json")
    action_dem_dicts = read_data_from_json(f"env/{env_config}/action_dem_dicts.json")
    action_order_dicts = read_data_from_json(f"env/{env_config}/action_order_dicts.json")
    action_price_dicts = read_data_from_json(f"env/{env_config}/action_price_dicts.json")

    return action_order_dicts, action_sup_dicts, action_dem_dicts, action_price_dicts


# Create a multipartite graph
def draw_multipartite_graph(env, t: int, save_prefix: str):

    num_stages = env.num_stages
    max_num_agents_per_stage = env.max_num_agents_per_stage
    sup_rel = env.supply_relations
    dem_rel = env.demand_relations
    save_path = f'results/{save_prefix}/'
    running_agents = env.running_agents

    M = nx.DiGraph()

    # Add nodes for each set
    stage_agents = []
    for m in range(num_stages):
        stage_agents = []
        for x in range(max_num_agents_per_stage):
            if running_agents[m][x] == -1:
                continue
            stage_agents.append(f"s{m}a{x}")
        M.add_nodes_from(stage_agents, layer=num_stages-m)  # Add set A nodes

    # Add edges between the sets
    edges = []
    for m in range(num_stages-1):
        for x in range(max_num_agents_per_stage):
            if running_agents[m][x] == -1:
                continue
            for i in range(max_num_agents_per_stage):
                if running_agents[m+1][i] == -1:
                    continue
                if sup_rel[m][x][i] == 1:
                    src = f"s{m+1}a{i}"
                    tgt = f"s{m}a{x}"
                    edges.append((src, tgt))
    M.add_edges_from(edges)

    # Define positions for the multipartite layout
    pos = nx.multipartite_layout(M, subset_key="layer")

    # Draw the multipartite graph
    # stage_colors = plt.cm.plasma(np.linspace(0, 1, 4))
    stage_colors = ["gold", "violet", "limegreen", "darkorange", "red", "green", "black", "pink"]
    colors = [stage_colors[m] for m in range(num_stages) for x in range(max_num_agents_per_stage) if env.running_agents[m][x] > -1]
    # mask closed agents
    for m in range(num_stages):
        for x in range(max_num_agents_per_stage):
            if env.running_agents[m][x] == 0:
                colors[m*max_num_agents_per_stage+x] = "black"

    plt.figure(figsize=(15, 10))
    nx.draw(M, pos, with_labels=True, node_color=colors, node_size=200, font_size=12, edge_color="gray", alpha=1)
    plt.title("Multipartite Graph")
    plt.savefig(os.path.join(save_path, "img_results", f"supply_chain_period_{t}.jpg"), format="jpg")




def visualize_state(env, t: int, save_prefix: str):
    
    # env.update_state_on_t(env.period-1)
    state_dict = env.state_dict
    num_stages = env.num_stages
    max_num_agents_per_stage = env.max_num_agents_per_stage
    lt_max = env.max_lead_time
    save_path = f'results/{save_prefix}/'
    # xy_locations = np.load(f'env/{save_prefix}/xy_locations.npy')

    df = pd.DataFrame({
        "stage": {},
        "agent_idx": {}, 
        "profits": {}, 
        "prod_capacity": {},
        "inventory": {},
        "sale_price": {},       
        "backlog_cost": {},
        "holding_cost": {},
        "backlog": {}, 
        "upstream_backlog": {},
        "suppliers": {},
        "customers": {},
        "order_cost": {},
        "prod_cost": {},
        "recent_sales": {},
        "lead_time": {},
        "deliveries": {},
        "orders": {},
        'running_status': {},
        "demand": {},
        "location": {},
        "supply_relation_summary": {},
    })
    for stage in range(num_stages):
        for agent in range(max_num_agents_per_stage):
            if env.running_agents[stage][agent] == -1:
                continue
            if stage == 0:
                demand = env.demands[agent, env.period]
            else:
                demand = 0
            if stage < num_stages-1: 
                num_avail_upstream = sum(env.running_agents[stage+1]>=0)
            else:
                num_avail_upstream = sum(env.running_agents[-1]>=0) # num of supplier == num of manufacturer
            supply_relation_summary = np.zeros(num_avail_upstream, dtype=int)
            next_supplier = np.argmax(state_dict[f'stage_{stage}_agent_{agent}'][9]) # supply_relation
            print("next supplier is", next_supplier)
            supply_relation_summary[next_supplier] = 2
            cur_supplier = [i for i in range(num_avail_upstream) if state_dict[f'stage_{stage}_agent_{agent}'][15][i]==1] # order
            print("current supplier is", cur_supplier)
            supply_relation_summary[cur_supplier] = 1
            total_deliveries = np.sum(state_dict[f'stage_{stage}_agent_{agent}'][12], axis=-1)
            cur_delivery = [i for i in range(num_avail_upstream) if total_deliveries[i]> 0] # agents that have deliveries on the way
            print("cur delivery is", cur_delivery)
            supply_relation_summary[cur_delivery] = 1
            print("supply relation summary", supply_relation_summary)

            df = pd.concat([df, pd.DataFrame({
                'stage': [stage], 
                "agent_idx": [agent],
                "prod_capacity": [state_dict[f'stage_{stage}_agent_{agent}'][0]],
                'sale_price': [state_dict[f'stage_{stage}_agent_{agent}'][1]],
                'order_cost': [state_dict[f'stage_{stage}_agent_{agent}'][2]],
                'backlog_cost': [state_dict[f'stage_{stage}_agent_{agent}'][3]],
                'holding_cost': [state_dict[f'stage_{stage}_agent_{agent}'][4]],
                'lead_time': [state_dict[f'stage_{stage}_agent_{agent}'][5]],
                'inventory': [state_dict[f'stage_{stage}_agent_{agent}'][6]],
                'backlog': [state_dict[f'stage_{stage}_agent_{agent}'][7]],
                'upstream_backlog': [state_dict[f'stage_{stage}_agent_{agent}'][8]],
                "suppliers": [state_dict[f'stage_{stage}_agent_{agent}'][9]],
                "customers": [state_dict[f'stage_{stage}_agent_{agent}'][10]],
                'recent_sales': [state_dict[f'stage_{stage}_agent_{agent}'][11]],
                'deliveries': [state_dict[f'stage_{stage}_agent_{agent}'][12]],
                'prod_cost': [state_dict[f'stage_{stage}_agent_{agent}'][13]], 
                'running_status': [state_dict[f'stage_{stage}_agent_{agent}'][14]],
                "orders": [state_dict[f'stage_{stage}_agent_{agent}'][15]],
                'profits': [state_dict[f'stage_{stage}_agent_{agent}'][16]],
                "demand": [demand],
                "location": [state_dict[f"stage_{stage}_agent_{agent}"][17]],
                "supply_relation_summary": [supply_relation_summary.tolist()], 
                })], ignore_index=True)
    
    df = df.groupby(by=['stage', 'agent_idx']).apply(lambda x: x).reset_index(drop=True)
    df['stage'] =df['stage'].astype(int)
    df['agent_idx'] = df['agent_idx'].astype(int)
    df['running_status'] = df['running_status'].astype(int)
    df['profits'] = df['profits'].astype(int)
    df['prod_capacity'] = df['prod_capacity'].astype(int)
    df['inventory'] = df['inventory'].astype(int)
    df['sale_price'] = df['sale_price'].astype(int)
    df['backlog_cost'] = df['backlog_cost'].astype(int)
    df['holding_cost'] = df['holding_cost'].astype(int)
    df['backlog'] = df['backlog'].astype(int)
    df['demand'] = df['demand'].astype(int)
    # df['upstream_backlog'] = df['upstream_backlog'].astype(int)
    # df['order_cost'] = df['order_cost'].astype(int)
    df['prod_cost'] = df['prod_cost'].astype(int)

    print("save data to", os.path.join(save_path, "df_results", f"env_period_{t}.csv"))
    df.to_csv(os.path.join(save_path, "df_results", f"env_period_{t}.csv"), index=False)
    df.to_json(os.path.join(save_path, "json_results", f"env_period_{t}.json"), orient='records', indent=4)
    draw_multipartite_graph(env=env, t=t, save_prefix=save_prefix)
    # draw_material_flow(env=env, t=t, save_prefix=save_prefix)
    return df.to_json(orient='records', indent=4)


def random_relations(n_cand: int, n_relation: int):

    return np.random.choice(a=n_cand, size=n_relation, replace=False)


def get_state_description(state: dict, past_req_orders: list, G: nx.Graph, state_format: str, enable_graph_change: bool, agent_name: str=None):
    if state_format == 'base':
        return get_base_description(state=state, past_req_orders=past_req_orders)
    elif state_format == "GraphML":
        return get_GraphML_description(G=G, agent_name=agent_name, enable_graph_change=enable_graph_change, state=state)
    else:
        raise AssertionError(f"{state_format} state description method not implemented yet")


def get_GraphML_description(agent_name: str, G: nx.DiGraph, enable_graph_change: bool, state: dict):

    # Convert to GraphML format
    # print(G.nodes())
    # print(G.edges())
    if enable_graph_change:
        upstream_nodes = [up for up in G.successors(agent_name) if G.nodes[up].get("stage")==G.nodes[agent_name].get("stage")+1]
        customer_nodes = [customer for node, customer in G.edges(agent_name) if G.edges[node, customer].get("supplier")]
        connected_nodes = upstream_nodes + customer_nodes + [agent_name]
        sub_graph = G.subgraph(connected_nodes)
    else:
        # Get nodes that have a "suppliers" relation with the given agent
        supplier_nodes = [supplier for node, supplier in G.edges(agent_name) if G.edges[node, supplier].get('customer')]
        customer_nodes = [customer for node, customer in G.edges(agent_name) if G.edges[node, customer].get("supplier")]
        sub_graph = G.subgraph(supplier_nodes + customer_nodes + [agent_name])
    graphml_str = '\n'.join(list(nx.generate_graphml(sub_graph, named_key_ids=True, prettyprint=True))[12:])

    recent_sales = f"\nPrevious Sales (in the recent round(s), from old to new): {state['sales']}\n"
    return graphml_str + recent_sales


def get_base_description(state, past_req_orders):

    suppliers = "; ".join([f"agent{i}" for i, _ in enumerate(state['suppliers']) if state['suppliers'][i]==1])
    non_suppliers = "; ".join([f"agent{i}" for i, _ in enumerate(state['suppliers']) if state['suppliers'][i]==0])
    lead_times = " round(s); ".join([f"from agent{i}: {state['lead_times'][i]}" for i, _ in enumerate(state['lead_times'])])
    order_costs = " unit(s); ".join([f"from agent{i}: {state['order_costs'][i]}" for i, _ in enumerate(state['order_costs'])])
    prod_cost = state["prod_cost"]
    # get the arriving deliveries from the upstream in this round
    arriving_delieveries = []
    for i, _ in enumerate(state['suppliers']):
        if state['suppliers'][i] == 1:
            arriving_delieveries.append(f"from agent{i}: {state['deliveries'][i][-state['lead_times'][i]:]}")
    arriving_delieveries = "; ".join(arriving_delieveries)

    # get the requested orders from downstreams in this round
    req_orders = []
    if len(past_req_orders) == 0:
        req_orders = "None"
    else:
        for i, _ in enumerate(past_req_orders):
            if past_req_orders[i] != 0:
                req_orders.append(f"from agent{i}: {past_req_orders[i]}")
        req_orders = " ".join(req_orders)
    # print("req orders", req_orders)

    return (
        f" - Lead Time: {lead_times} round(s)\n"
        f" - Order costs: {order_costs} unit(s)\n"
        f" - Production costs: {prod_cost} unit(s)\n"
        f" - Inventory Level: {state['inventory']} unit(s)\n"
        f" - Production capacity: {state['prod_capacity']} unit(s)\n"
        f" - Current Backlog (you owing to the downstream): {state['backlog']} unit(s)\n"
        f" - Upstream Backlog (your upstream owing to you): {state['upstream_backlog']} unit(s)\n"
        f" - Previous Sales (in the recent round(s), from old to new): {state['sales']}\n"
        f" - In the last round, you placed orders to upstream suppliers: {req_orders}\n"
        f" - Arriving Deliveries (in this and the next round(s), from near to far): {arriving_delieveries}\n"
        f" - Your upstream suppliers are: {suppliers}\n" 
        f" - Other available upstream agents in the environment are: {non_suppliers}\n"
    )


def get_demand_description(demand_fn: Callable) -> str:
    
    if demand_fn.dist == "constant_demand":
        mean = demand_fn.mean
        return f"The expected demand at the retailer (stage 0) is a constant {mean} units for all rounds."
    elif demand_fn.dist == "uniform_demand":
        lb = demand_fn.lb
        ub = demand_fn.ub
        return f"The expected demand at the retailer (stage 0) is a discrete uniform distribution U{lb, ub} for all rounds."
    elif demand_fn.dist == "seasonal_demand":
        return f"The expected demand at the retailer (stage 0) is a discrete uniform distribution U{0, 4} for the first 4 rounds, " \
            "and a discrete uniform distribution U{5, 8} for the last 8 rounds."
    elif demand_fn.dist == "normal_demand":
        mu = demand_fn.mean
        std = demand_fn.std
        return f"The expected demand at the retailer (stage 0) is a normal distribution N({mu}, {std}), " \
            "truncated at 0, for all 12 rounds."
    elif demand_fn.dist == "dyn_poisson_demand":
        mean = demand_fn.mean
        return f"The expected demand at the retailer (stage 0) is a poisson distribution P(lambda={mean}), and the lambda is increasingly bigger."
    else:
        raise KeyError(f"Error: {demand_fn} not implemented.")
  


def update_sup_action(sup_action: list, rm_match: str, add_match: str):
    
    remove_sup = rm_match.replace(" ", "")                
    if remove_sup != "":
        remove_sup = remove_sup.replace("agent", "").replace('"', "")
        try:
            remove_sup = [int(ind) for ind in remove_sup.split(",")]
            for ind in remove_sup:
                sup_action[ind] = 0
        except: # if the string format is invalid
            pass
    add_sup = add_match.replace(" ", "")   
    if add_sup != "":
        add_sup = add_sup.replace("agent", "").replace('"', "")
        try:
            add_sup = [int(ind) for ind in add_sup.split(",")]
            for ind in add_sup:
                sup_action[ind] = 1
        except:
            pass
    
    return sup_action
    
def stage_agent_id2name(txt: str):
    """
    Create a mapping of agent IDs to their names for different stages.
    
    Returns:
        dict: A dictionary mapping agent IDs to their names.
    """
    txt = txt.replace("stage_0_agent_", "retailer_")
    txt = txt.replace("stage_1_agent_", "wholesaler_")
    txt = txt.replace("stage_2_agent_", "distributor_")
    txt = txt.replace("stage_3_agent_", "manufacturer_")

    return txt 
   
def name2stage_agent_id(txt: str):

    txt = txt.replace("retailer ", "stage_0_agent_")
    txt = txt.replace("wholesaler ", "stage_1_agent_")
    txt = txt.replace("distributor ", "stage_2_agent_")
    txt = txt.replace("manufacturer ", "stage_3_agent_")

    return txt