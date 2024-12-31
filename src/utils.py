import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

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

def generate_lead_time(num_data: int, lb=2, ub=8):

    return np.random.uniform(low=lb, high=ub, size=num_data)
