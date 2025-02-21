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
sys.path.append('/home/vislab/Yanjia/MAS_SupplyChain')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.config import env_configs_list, get_env_configs
from model.utils.utils import clear_dir, split_demand, save_data_to_json, read_data_from_json
from src.model.data_simulation import generate_lead_time, generate_prod_capacity
from src.model.data_simulation import generate_cost_price, generate_sup_dem_relations
from src.model.data_simulation import generate_holding_costs, generate_backlog_costs, generate_init_inventories
from src.model.data_simulation import Demand_fn
import matplotlib.pyplot as plt
import networkx as nx
import random
import csv
from tqdm import tqdm
import torch


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
    ["Increased competition", "Negative", ["Price"]],
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
    
def visualize_contextualized_supply_chain(env: dict, event_dict: dict, df_edges: pd.DataFrame, df_nodes: pd.DataFrame):
    
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
    plt.show()

# %%

def assign_events(num_events: int, num_stages: int, num_agents_per_stage: int):
    num_current_events = random.choice(range(1, 4))
    # print("assign ", num_current_events, " events")
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
    df_node = pd.DataFrame(index=range(num_nodes), columns=["node_id", "node_attr", "type"])
    df_node["node_id"] = np.arange(num_nodes)
    df_node["node_attr"] = [f"stage_{m}_agent_{x}" for m in range(num_stages) for x in range(num_agents_per_stage)] + [event_dict['events'][eidx] for eidx in env['events'].keys()]
    df_node["type"] = [stage_names[m] for m in range(num_stages) for x in range(num_agents_per_stage)] + ["event" for _ in range(num_current_events)]
    
    return df_node

# %%
def convert_env_to_edge_df(env: dict, event_dict: dict):
    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    sup_rel = env['supply_relations']
    num_edges = sum([sum([sum(sup_rel[m][x]) for x in range(num_agents_per_stage)]) for m in range(num_stages-1)]) + len(env['events'])
    df_edge = pd.DataFrame(index=range(num_edges), columns=["source", "target", "label", "type", 'aspect'])
    edge_idx = 0
    # Randomly create backlog events between suppliers and customers
    for m in range(num_stages-1):
        for x in range(num_agents_per_stage):
            for i in range(num_agents_per_stage):
                if sup_rel[m][x][i] == 1:
                    num_request_order = random.choice(range(5, 50))
                    is_fulfilled = random.choice([False, True])
                    if is_fulfilled:
                        df_edge.loc[edge_idx, ["source", "target", "label", "type", 'aspect']] = \
                            [f"stage_{m}_agent_{i}", f"stage_{m+1}_agent_{x}", f"request order of {num_request_order} units of product", "", []]
                        df_edge.loc[edge_idx+1, ["source", "target", "label", 'type', 'aspect']] = \
                            [f"stage_{m+1}_agent_{x}", f"stage_{m}_agent_{i}", f"deliverying {num_request_order} units of product", "Positive", ['Order Fulfillment']]
                    else: # Not fulfilled
                        num_delivery = random.choice(range(num_request_order))
                        df_edge.loc[edge_idx, ["source", "target", "label", 'type', 'aspect']] = \
                            [f"stage_{m}_agent_{i}", f"stage_{m+1}_agent_{x}", f"request order of {num_request_order} units of product", '', []]
                        df_edge.loc[edge_idx+1, ["source", "target", "label", 'type', 'aspect']] = \
                            [f"stage_{m+1}_agent_{x}", f"stage_{m}_agent_{i}", f"deliverying {num_delivery} units of product", 'Negative', ['Order Fulfillment']]
                    edge_idx += 2

    # Keep the record of the other events
    for eidx in env['events'].keys():
        event_name = event_dict['events'][eidx]
        stage_idx, agent_idx = env['events'][eidx]
        event_type = event_dict['Type'][eidx]
        aspect = event_dict['Aspect'][eidx]
        df_edge.loc[edge_idx, ["source", "target", "label", 'type', 'aspect']] = \
            [event_name, f"stage_{stage_idx}_agent_{agent_idx}", "affects", event_type, aspect]
        edge_idx += 1

    # Keep the record of the supply relations
    for m in range(num_stages-1):
        for x in range(num_agents_per_stage):
            for i in range(num_agents_per_stage):
                if sup_rel[m][x][i] == 1:
                    df_edge.loc[edge_idx, ["source", "target", "label", 'type', 'aspect']] = \
                        [f"stage_{m+1}_agent_{i}", f"stage_{m}_agent_{x}", "is supplier", "", []]
                    edge_idx += 1
                else:
                    pass
                    # df_edge.loc[edge_idx, ["source", "target", "label", 'type', 'aspect']] = \
                    #     [f"stage_{m+1}_agent_{i}", f"stage_{m}_agent_{x}", "potential supplier", '', []]

    # # create a column named "is_backlog" to indicate whether the edge is related to backlog
    # df_edge['is_backlog'] = df_edge['aspect'].apply(lambda x: 'Backlog' in x)
    # # create a column named "is_price" to indicate whether the edge is related to price
    # df_edge['is_price'] = df_edge['aspect'].apply(lambda x: 'Price' in x)
    # # create a column named "is_demand" to indicate whether the edge is related to demand
    # df_edge['is_demand'] = df_edge['aspect'].apply(lambda x: 'Demand' in x)
    # # create a column named "is_delivery_time" to indicate whether the edge is related to delivery time
    # df_edge['is_delivery_time'] = df_edge['aspect'].apply(lambda x: 'Delivery Time' in x)
    # # create a column named "is_production" to indicate whether the edge is related to production
    # df_edge['is_production_capacity'] = df_edge['aspect'].apply(lambda x: 'Production Capacity' in x)
    
    return df_edge

def build_event_graph(df_edges: pd.DataFrame, df_nodes: pd.DataFrame):
    G = nx.DiGraph()
    for i in range(len(df_nodes)):
        G.add_node(df_nodes['node_attr'][i], type=df_nodes['type'][i])
    for i in range(len(df_edges)):
        G.add_edge(df_edges['source'][i], df_edges['target'][i], label=df_edges['label'][i], type=df_edges['type'][i], aspect=df_edges['aspect'][i])
    return G


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
    # profit_rates = \
    #     generate_profit_rates(dist=env_configs["profit_rate_dist"], num_data=num_total_agents, config_name=env_configs["config_name"])

    demand_fn = Demand_fn(dist=env_configs["demand_fn"]['dist'], mean=env_configs["demand_fn"].get("mean", 0), std=env_configs["demand_fn"].get("std", 0), 
                            lb=env_configs["demand_fn"].get("lb", 0), ub=env_configs["demand_fn"].get("ub", 0), trend=env_configs["demand_fn"].get("trend", False))
    stage_names = env_configs["stage_names"]

    return {
            'num_stages': num_stages,
            'num_periods': num_periods,
            'num_agents_per_stage': num_agents_per_stage,
            "demand_dist": env_configs["demand_fn"]["dist"],
            'init_inventories': init_inventories, # num_stages * num_agents_per_stage
            'lead_times': lead_times, # num_stages * num_agents_per_stage * num_agents_per_stage
            'demand_fn': demand_fn,
            'prod_capacities': prod_capacities,
            'sale_prices': sale_prices,
            'order_costs': order_costs,
            "prod_costs": prod_costs, 
            'backlog_costs': backlog_costs,
            'holding_costs': holding_costs,
            'num_init_suppliers': num_init_suppliers, 
            'supply_relations': supply_relations,
            "demand_relations": demand_relations,
            'stage_names': stage_names,
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

def generate_orderFulfill_questions(target_node:str, event_node:str, event_type: str):

    question = f"Your are {target_node}. Based on the provided supply chain graph, how is the performance of your supplier {event_node} in terms of order fulfillment? Answer either 'positive', 'negative'."
    answer = "positive" if event_type == "Positive" else "negative"

    return question, answer

def generate_questions(df_edges: pd.DataFrame, env: dict, num_questions:int=20):

    
    num_stages = env['num_stages']
    num_agents_per_stage = env['num_agents_per_stage']
    aspect_list = ["Production Capacity", "Delivery Time", "Order Fulfillment", "Price", "Demand"]

    # make a list of list to list
    event_aspect_in_graph = []
    for x in df_edges['aspect']:
        event_aspect_in_graph += x
    event_aspect_in_graph = list(set(event_aspect_in_graph)) + ["Order Fulfillment"]
    # event_aspect_in_graph + ["Order Fulfillment" for _ in event_aspect_in_graph] # To balance the question ratio
    num_event_aspect_in_graph = len(event_aspect_in_graph)
    
    # remove price from the aspect list
    questions = []
    answers = []
    for i in range(num_questions):
        # Get a valid event
        es = event_aspect_in_graph[i%num_event_aspect_in_graph]
        df_event_aspect = df_edges[df_edges['aspect'].apply(lambda x: es in x)].reset_index(drop=True)
        row_id = random.choice(range(len(df_event_aspect)))
        event_aspect = df_event_aspect.loc[row_id, 'aspect']

        event_type = df_event_aspect.loc[row_id, 'type']
        event_target_node = df_event_aspect.loc[row_id, 'target']
        event_node = df_event_aspect.loc[row_id, 'source']


        # Generate a target node to ask about
        target_node_stage_id, target_node_agent_id = generate_target_node(num_stages=num_stages, num_agents_per_stage=num_agents_per_stage)
        # if it is the same as the event node, regenerate the target node
        while int(event_target_node.split("_")[1]) == target_node_stage_id and int(event_target_node.split("_")[3]) == target_node_agent_id:
            target_node_stage_id, target_node_agent_id = generate_target_node(num_stages=num_stages, num_agents_per_stage=num_agents_per_stage)
        target_node = f"stage_{target_node_stage_id}_agent_{target_node_agent_id}"

        if int(event_target_node.split("_")[1]) < target_node_stage_id: 
            event_target_node_relation = "customer"  
        elif int(event_target_node.split("_")[1]) > target_node_stage_id:
            event_target_node_relation = "supplier"
        else:
            event_target_node_relation = "competitor"

        # Special case to deal with backlog judgement
        if "Order Fulfillment" in event_aspect:
            question, answer = generate_orderFulfill_questions(target_node=event_target_node, event_node=event_node, event_type=event_type)
            questions.append(question)
            answers.append(answer)
        # Other events
        elif check_connection(G=G, event_target_node=event_target_node, target_node=event_target_node):
            asp = random.choice(aspect_list)
            node_rel = random.choice(['upstream suppliers reliability', "downstream customers demand"])
            questions.append(f"Your are {target_node}. Based on the provided supply chain graph, how would the {event_node} affect any of your {node_rel} in terms of {asp}? Answer either 'positive' or 'negative' if it happens to your supplier(s), otherwise answer'neutral'.")
            if asp in event_aspect and event_target_node_relation == node_rel:
                answers.append("positive" if event_type == "Positive" else "negative")
            else:
                answers.append("neutral")
        else:
            asp = random.choice(aspect_list)
            node_rel = random.choice(['upstream suppliers reliability', "downstream customers demand"])
            questions.append(f"Your are {target_node}. Based on the provided supply chain graph, how would the {event_node} affect any of your {node_rel} in terms of {asp}? Answer either 'positive' or 'negative' if it happens to your supplier(s), otherwise answer'neutral'.")
            answers.append("neutral")

    return questions, answers





if __name__ == "__main__":

    num_graphs = 3000
    num_questions_per_graph = 10
    events_list = read_data_from_json(read_path="src/gnn/gnn_dataset/supply_chain_events.json")
    event_dict = {"events": [x[0] for x in events_list[1:]],
                "Type": [x[1] for x in events_list[1:]],
                "Aspect": [x[2] for x in events_list[1:]]}
    num_events = len(event_dict['events'])


    env_config_name = "large_graph_test"
    save_path = f"src/gnn/gnn_dataset"
    os.makedirs(f"{save_path}/{env_config_name}/nodes", exist_ok=True)
    clear_dir(f"{save_path}/{env_config_name}/nodes")
    os.makedirs(f"{save_path}/{env_config_name}/edges", exist_ok=True)
    clear_dir(f"{save_path}/{env_config_name}/edges")
    os.makedirs(f"{save_path}/{env_config_name}/graphs", exist_ok=True)
    clear_dir(f"{save_path}/{env_config_name}/graphs")

    df_qa = pd.DataFrame({"question": [], "label": []})

    for data_idx in tqdm(range(num_graphs)):
        env = generate_env(env_config_name=env_config_name)
        num_stages = env['num_stages']
        num_agents_per_stage = env['num_agents_per_stage']
        # %%
        events = assign_events(num_events, num_stages, num_agents_per_stage)
        env['events'] = events
        df_nodes = convert_env_to_node_df(env=env)
        df_edges = convert_env_to_edge_df(env=env, event_dict=event_dict)


        # %%
        G = build_event_graph(df_edges=df_edges, df_nodes=df_nodes)

        questions, answers = generate_questions(df_edges=df_edges, env=env, num_questions=num_questions_per_graph)
        # save questions and answers to csv
        df_qa = pd.concat([df_qa, pd.DataFrame({"question": questions, "label": answers})], axis=0)
        
        # %%
        # visualize_contextualized_supply_chain(env=env, event_dict=event_dict, df_edges=df_edges, df_nodes=df_nodes)

        # %%
        # node_id,node_attr
        # 0,cannabis
        # 1,marijuana
        # 2,legal
        # 3,more available
        # 4,good thing

        df_nodes.head()
        for i in range(num_questions_per_graph):
            df_nodes.to_csv(f"{save_path}/{env_config_name}/nodes/{data_idx*num_questions_per_graph+i}.csv", index=False)

        # %%
        # src,edge_attr,dst
        # 0,synonym of,1
        # 2,causes,3
        # 1,capable of,4
        # 4,desires,2
        node_id_name_map = dict(zip(df_nodes['node_attr'].tolist(), df_nodes['node_id'].tolist()))
        df_simp_edges = pd.DataFrame({
            "src": [node_id_name_map[x] for x in df_edges['source']],
            "edge_attr": df_edges['label'],
            "dst": [node_id_name_map[x] for x in df_edges['target']]
            })
        for i in range(num_questions_per_graph):
            df_simp_edges.to_csv(f"{save_path}/{env_config_name}/edges/{data_idx*num_questions_per_graph+i}.csv", index=False)

    df_qa.to_csv(f"{save_path}/{env_config_name}/all_questions.csv", index=False)




