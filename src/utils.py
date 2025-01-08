import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import json

def save_string_to_file(data: str, save_path: str, t: int):
    print("Saving data to: ", f"env/{save_path}/chat_summary_period{t}.txt")
    with open(f"results/{save_path}/chat_summary_period{t}.txt", 'w') as f:
        f.write(data)

def save_dict_to_json(data: dict, save_path: str):
    print("Saving config to: ", f"env/{save_path}/config.json")
    with open(f"env/{save_path}/config.json", 'w') as f:
        json.dump(data, f)

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
                    "suppliers": [state_dict[f'stage_{stage}_agent_{agent}'][9]],
                    "customers": [state_dict[f'stage_{stage}_agent_{agent}'][10]],
                    'recent_sales': [state_dict[f'stage_{stage}_agent_{agent}'][11]],
                    'deliveries': [state_dict[f'stage_{stage}_agent_{agent}'][12]],
                    'profits': [rewards.get(f'stage_{stage}_agent_{agent}', None)]
                    })], ignore_index=True)
            
    df = df.groupby(by=['stage', 'agent_idx']).apply(lambda x: x).reset_index(drop=True)

    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, f"env_period_{t}.csv"), index=False)
    draw_multipartite_graph(env=env, t=t, save_prefix=save_prefix)


def random_relations(n_cand: int, n_relation: int):

    return np.random.choice(a=n_cand, size=n_relation, replace=False)


def get_state_description(state, state_idx: int):

    suppliers = "; ".join([f"agent{i}" for i, _ in enumerate(state['suppliers']) if state['suppliers'][i]==1])
    non_suppliers = "; ".join([f"agent{i}" for i, _ in enumerate(state['suppliers']) if state['suppliers'][i]==0])
    lead_times = " round(s); ".join([f"from agent{i}: {state['lead_times'][i]}" for i, _ in enumerate(state['lead_times'])])
    arriving_delieveries = []
    for i, _ in enumerate(state['suppliers']):
        if state['suppliers'][i] == 1:
            arriving_delieveries.append(f"from agent{i}: {state['deliveries'][i][-state['lead_times'][i]:]}")
    order_costs = " unit(s); ".join([f"from agent{i}: {state['order_costs'][i]}" for i, _ in enumerate(state['order_costs'])])

    arriving_delieveries = " ".join(arriving_delieveries)

    return (
        f" - Lead Time: {lead_times} round(s)\n"
        f" - Order costs: {order_costs} unit(s)\n"
        f" - Inventory Level: {state['inventory']} unit(s)\n"
        f" - Current Backlog (you owing to the downstream): {state['backlog']} unit(s)\n"
        f" - Upstream Backlog (your upstream owing to you): {state['upstream_backlog']} unit(s)\n"
        f" - Previous Sales (in the recent round(s), from old to new): {state['sales']}\n"
        f" - Arriving Deliveries (in this and the next round(s), from near to far): {arriving_delieveries}\n"
        f" - Your upstream suppliers are: {suppliers}\n" 
        f" - Other upstream suppliers are: {non_suppliers}\n"
    )


def get_demand_description(demand_fn: str) -> str:
    if demand_fn == "constant_demand":
        return "The expected demand at the retailer (stage 1) is a constant 4 units for all rounds."
    elif demand_fn == "uniform_demand":
        return "The expected demand at the retailer (stage 1) is a discrete uniform distribution U{0, 4} for all rounds."
    elif demand_fn == "larger_demand":
        return "The expected demand at the retailer (stage 1) is a discrete uniform distribution U{0, 8} for all rounds."
    elif demand_fn == "seasonal_demand":
        return "The expected demand at the retailer (stage 1) is a discrete uniform distribution U{0, 4} for the first 4 rounds, " \
            "and a discrete uniform distribution U{5, 8} for the last 8 rounds."
    elif demand_fn == "normal_demand":
        return "The expected demand at the retailer (stage 1) is a normal distribution N(4, 2^2), " \
            "truncated at 0, for all 12 rounds."
    else:
        raise KeyError(f"Error: {demand_fn} not implemented.")
  