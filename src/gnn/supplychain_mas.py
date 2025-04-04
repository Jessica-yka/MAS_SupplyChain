import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data.data import Data
import sys
sys.path.append('/data/yanjia/MAS_SupplyChain')
from src.gnn.preprocess.lm_modeling import load_model, load_text2embedding
from src.gnn.preprocess.graph_qa_data_generation import build_supplier_graph
from src.gnn.preprocess.utils.retrieval import get_sub_df_nodes, get_sub_df_edges
from src.gnn.preprocess.utils.retrieval import get_demand_sub_df_edges, get_event_sub_df_edges, get_lt_sub_df_edges, get_price_sub_df_edges, get_of_sub_df_edges
from src.model.utils.utils import name2stage_agent_id
import os
from tqdm import tqdm
from src.gnn.preprocess.events import events
import csv


PATH = 'src/gnn/gnn_dataset/large_graph_test/test_data'

path_nodes = f'{PATH}/nodes'
path_edges = f'{PATH}/edges'
path_graphs = f'{PATH}/graphs'

cached_graph = f'{PATH}/cached_graphs'
cached_desc = f'{PATH}/cached_desc'

# the dataloader to test gnn-enhanced llama on downstream tasks. Can be removed from the final version of the codebase
class SupplyChainMASTestDataset(Dataset):
    def __init__(self, event_dataset: str=None, supplier_dataset: str=None, price_dataset: str=None, lead_time_dataset: str=None, cot: bool = False):
        super().__init__()

        self.use_event = False
        self.use_supplier = False
        self.use_price = False
        self.use_lead_time = False
        if cot:
            self.cot = "Let's think step by step."
        if event_dataset: 
            self.event_text = pd.read_csv(f'{PATH}/{event_dataset}')
            self.num_data = len(self.event_text)
            self.use_event = True
        if supplier_dataset:
            self.supplier_text = pd.read_csv(f'{PATH}/{supplier_dataset}')
            self.num_data = len(self.supplier_text)
            self.use_supplier = True
        if price_dataset:
            self.price_text = pd.read_csv(f'{PATH}/{price_dataset}')
            self.num_data = len(self.price_text)
            self.use_price = True
        if lead_time_dataset:
            self.lead_time_text = pd.read_csv(f'{PATH}/{lead_time_dataset}')
            self.num_data = len(self.lead_time_text)
            self.use_lead_time = True

        # self.prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.graph = None
        self.graph_type = 'Contextualized Supply Chain Graph'
        self.type = 'mas_qa'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        data = {}
        data['id'] = {'id': index}
        if self.use_event:
            question = self.event_text.loc[index, 'question']
            label = self.event_text.loc[index, 'label']
            data_index = int(self.event_text.loc[index, 'graph_idx'])
            graph = torch.load(f'{cached_graph}/{data_index}.pt')
            desc = open(f'{cached_desc}/{data_index}.txt', 'r').read()
            data['event_data'] = {
                                    'id': data_index,
                                    'label': label,
                                    'desc': desc,
                                    'graph': graph,
                                    'question': question
                                }
        # else:
        #     data['event_data'] = None

        if self.use_lead_time:
            question = self.lead_time_text.loc[index, 'question']
            label = self.lead_time_text.loc[index, 'label']
            data_index = int(self.lead_time_text.loc[index, 'graph_idx'])
            graph = torch.load(f'{cached_graph}/{data_index}.pt')
            desc = open(f'{cached_desc}/{data_index}.txt', 'r').read()
            data['lead_time_data'] = {
                                        'id': data_index,
                                        'label': label,
                                        'desc': desc,
                                        'graph': graph,
                                        'question': question
                                    }
        # else:
        #     data['lead_time_data'] = None

        if self.use_price:
            question = self.price_text.loc[index, 'question']
            label = self.price_text.loc[index, 'label']
            data_index = int(self.price_text.loc[index, 'graph_idx'])
            graph = torch.load(f'{cached_graph}/{data_index}.pt')
            desc = open(f'{cached_desc}/{data_index}.txt', 'r').read()
            data['price_data'] = {
                                    'id': data_index,
                                    'label': label,
                                    'desc': desc,
                                    'graph': graph,
                                    'question': question
                                }
        # else:
        #     data['price_data'] = None

        if self.use_supplier:
            question = self.supplier_text.loc[index, 'question']
            label = self.supplier_text.loc[index, 'label']
            data_index = int(self.supplier_text.loc[index, 'graph_idx'])
            graph = torch.load(f'{cached_graph}/{data_index}.pt')
            desc = open(f'{cached_desc}/{data_index}.txt', 'r').read()
            data['supplier_data'] = {
                                        'id': data_index,
                                        'label': label,
                                        'desc': desc,
                                        'graph': graph,
                                        'question': question
                                    }
        # else:
        #     data['supplier_data'] = None

        return data

    def get_idx_split(self):

        return {"test": range(self.num_data)}


# the dataloader used in the MAS system) (IMPORTANT)
class SupplyChainMASDataset(Dataset):
    # the stage_state contains all agents at the same stage.
    # since they are independent from each other. We can do calculation in parallel.
    def __init__(self, env): 
        super().__init__()
        
        # self.prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.graph = None
        self.graph_type = 'Contextualized Supply Chain Graph in the multi-agent system'
        self.type = 'sc_mas'
        self.num_stages = env.num_stages
        self.max_num_agents_per_stage = env.max_num_agents_per_stage
        self.env = env
        self.event_dict = {"events": [x[0] for x in events[1:]],
                            "Type": [x[1] for x in events[1:]],
                            "Aspect": [x[2] for x in events[1:]],
                            "id": [x[3] for x in events[1:]],
                            }
        self.index_lm_model = 'sbert'
        self.index = 0

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)
    
    def retrieve_env(self, env):

        return {
            'num_stages': env.num_stages,
            'demands': env.demands,
            't': env.period,
            'max_num_agents_per_stage': env.max_num_agents_per_stage,
            'inventories': env.inventories[:, :, env.period], # num_stages * max_num_agents_per_stage
            'lead_times': env.lead_times, # num_stages * max_num_agents_per_stage * max_num_agents_per_stage
            'prod_capacities': env.prod_capacities,
            'arriving_orders': env.arriving_orders,
            'sale_prices': env.sale_prices,
            'order_costs': env.order_costs,
            "prod_costs": env.prod_costs, 
            'backlog_costs': env.backlog_costs,
            'backlog': env.backlogs[:, :, env.period],
            'holding_costs': env.holding_costs,
            'supply_relations': env.supply_relations,
            "demand_relations": env.demand_relations,
            'stage_names': env.stage_names,
            'orders': env.orders, # because each agent has only one supplier, so l reduce the 3d array to 2d array
            "events": env.emergent_events,
            'running_agents': env.running_agents, 
        }

    def convert_env_to_node_df(self, env: dict, event_dict: dict):

        num_stages = env['num_stages']
        max_num_agents_per_stage = env['max_num_agents_per_stage']
        stage_names = env['stage_names']
        num_current_events = len(env['events'])
        # num_nodes = 1 + num_stages * max_num_agents_per_stage + num_current_events
        # df_node = pd.DataFrame(columns=["node_id", "type"])
        # df_node["node_id"] = np.arange(num_nodes).tolist()
        df_node = pd.DataFrame()
        df_node["name"] = ["Customers"] + [f"stage_{m}_agent_{x}" for m in range(num_stages) for x in range(max_num_agents_per_stage)] + [event_dict['events'][eidx] for eidx in env['events'].keys()]
        df_node["type"] = ["customers"] + [stage_names[m] for m in range(num_stages) for x in range(max_num_agents_per_stage)] + ["event" for _ in range(num_current_events)]
        df_node['sale_price'] = [0] + env['sale_prices'].flatten().tolist() + [0 for _ in range(num_current_events)]
        df_node['prod_capacity'] = [0] + env['prod_capacities'].flatten().tolist() + [0 for _ in range(num_current_events)]
        df_node['prod_cost'] = [0] + env['prod_costs'].flatten().tolist() + [0 for _ in range(num_current_events)]
        df_node['holding_cost'] = [0] + env['holding_costs'].flatten().tolist() + [0 for _ in range(num_current_events)]
        df_node['backlog_cost'] = [0] + env['backlog_costs'].flatten().tolist() + [0 for _ in range(num_current_events)]
        df_node['inventory'] = [0] + env['inventories'].flatten().tolist() + [0 for _ in range(num_current_events)]
        df_node['backlog'] = [0] + env['backlog'].flatten().tolist() + [0 for _ in range(num_current_events)]
        # df_node['upstream_backlog'] = [0] + env['upstream_backlog'].flatten().tolist() + [0 for _ in range(num_current_events)]
        df_node['stage_id'] = [-1] + [m for m in range(num_stages) for _ in range(max_num_agents_per_stage)] + [-1 for _ in env['events'].keys()]
        df_node['agent_id'] = [-1] + [x for _ in range(num_stages) for x in range(max_num_agents_per_stage)] + [-1 for _ in env['events'].keys()]
        df_node['running_status'] = [1] + env['running_agents'].flatten().tolist() + [1 for _ in range(num_current_events)]
        df_node = df_node[df_node['running_status'] == 1].reset_index(drop=True)
        df_node['node_id'] = df_node.index
        return df_node

    def convert_env_to_edge_df(self, env: dict, event_dict: dict, target_stage_idx: int, target_agent_idx: int):
        # TODO: check if the order is updated before the agents are queried.
        num_stages = env['num_stages']
        max_num_agents_per_stage = env['max_num_agents_per_stage']
        # num_init_suppliers = env['num_init_suppliers']
        sup_rel = env['supply_relations']
        # order_fulfill_rates = env['order_fulfill_rates']
        df_edge = pd.DataFrame(columns=["source", "target", "label", "type", 'aspect'])
        edge_idx = 0
        t = env['t']
        # create orders from the downstream agents
        # TODO: need to update demand
        if target_stage_idx == 0:
            num_unit = env['demands'][target_agent_idx, t]
            df_edge.loc[edge_idx, ["source", "target", "label", "type", "aspect"]] = \
                    ['Customers', f"stage_{target_stage_idx}_agent_{target_agent_idx}", f"order {num_unit} units of product at period {t} from", "", []]
            edge_idx += 1
        else:
            for customer_agent_idx in range(max_num_agents_per_stage):
                num_unit = env['orders'][target_stage_idx-1][customer_agent_idx][target_agent_idx][t]
                if num_unit > 0:
                    df_edge.loc[edge_idx, ["source", "target", "label", "type", "aspect"]] = \
                        [f"stage_{target_stage_idx-1}_agent_{customer_agent_idx}", f"stage_{target_stage_idx}_agent_{target_agent_idx}", f"order {num_unit} units of product at period {t} from", "", []]
                    edge_idx += 1
        # record the order made by the target to the upstream
        for supplier_agent_idx in range(max_num_agents_per_stage):
            lt = env['lead_times'][target_stage_idx][target_agent_idx][supplier_agent_idx]
            for day in range(np.min([lt, t])):
                num_unit = env['orders'][target_stage_idx][target_agent_idx][supplier_agent_idx][day] # TODO: how to get the recent orders
                if num_unit > 0:
                    df_edge.loc[edge_idx, ["source", "target", "label", "type", "aspect"]] = \
                        [f"stage_{target_stage_idx}_agent_{target_agent_idx}", f"stage_{target_stage_idx+1}_agent_{supplier_agent_idx}", f"order {num_unit} units of product at period {day} from", "", []]
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
            for x in range(max_num_agents_per_stage):
                if env["running_agents"][m][x] != 1:
                    continue
                for i in range(max_num_agents_per_stage):
                    if env["running_agents"][m+1][i] != 1:
                        continue
                    if sup_rel[m][x][i] == 1:
                        df_edge.loc[edge_idx, ["source", "target", "label", 'type', 'aspect']] = \
                            [f"stage_{m+1}_agent_{i}", f"stage_{m}_agent_{x}", "is the supplier of", "", []]
                        edge_idx += 1

        # add arriving deliveries information
        if m < num_stages-1:
            for supp_idx in range(self.max_num_agents_per_stage):
                if env["running_agents"][m+1][supp_idx] != 1:
                    continue
                lt = env['lead_times'][target_stage_idx][target_agent_idx][supp_idx]
                if t > lt:
                    arriving_orders = env['arriving_orders'][target_stage_idx, target_agent_idx, supp_idx, (t - lt + 1):(t + 1)]
                elif t > 0:
                    arriving_orders = env['arriving_orders'][target_stage_idx, target_agent_idx, supp_idx, 1:(t+1)]
                if t > 0:
                    for day, num_units in enumerate(arriving_orders[::-1]):
                        if num_units > 0:
                            df_edge.loc[edge_idx, ["source", "target", "label", "type", "aspect"]] = \
                            [f"stage_{target_stage_idx+1}_agent_{supp_idx}", f"stage_{target_stage_idx}_agent_{target_agent_idx}", f"deliverying {num_units} units of product in {lt-day} days", "", []]
                            edge_idx += 1
       
        # add lead time info to the edge_df
        for stage_id in range(num_stages-1):
            for agent_id in range(max_num_agents_per_stage):
                if env['running_agents'][stage_id][agent_id] != 1:
                    continue
                for supp_idx in range(max_num_agents_per_stage):
                    if env["running_agents"][stage_id+1][supp_idx] != 1:
                        continue
                    lt = env['lead_times'][stage_id][agent_id][supp_idx]
                    df_edge.loc[edge_idx, ["source", "target", "label", "type", "aspect"]] = \
                            [f"stage_{stage_id+1}_agent_{supp_idx}", f"stage_{stage_id}_agent_{agent_id}", f"has lead time of {lt} days to", "", []]
                    edge_idx += 1


        return df_edge

    def retrieve_subgraph(self, df_nodes, df_edges, target_node, G, question_type):

        df_sub_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node)
        node_id_name_map = dict(zip(df_sub_nodes['node_id'].tolist(), df_sub_nodes['name'].tolist()))
        if question_type == "order placement":
            # only retrieve the relation with downstream agents from the df_edges
            df_lt_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
            df_demand_edges = get_demand_sub_df_edges(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node)
            # df_lt_edges = get_lt_sub_df_edges(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node)
            df_sub_edges = pd.concat([df_demand_edges, df_lt_edges], axis=0)
        elif question_type == "supplier selection":
            # only retrieve the relations that are 1.upstream agents from the df_edges, 2.the supplier relations, and 3.the event
            df_event_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
            df_price_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
            df_of_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
            for edge in G.edges(data=True):
                if 'affects' in edge[2].get('label', ''):       
                    df_event_edges = get_event_sub_df_edges(G=G, df_edges=df_edges, df_nodes=df_nodes, target_node=target_node)
            # df_event_edges = get_event_sub_df_edges(G=G, df_edges=df_edges, df_nodes=df_nodes, target_node=target_node)
            df_price_edges = get_price_sub_df_edges(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node)
            df_lt_edges = get_lt_sub_df_edges(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node)
            df_of_edges = get_of_sub_df_edges(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node)
            df_sub_edges = pd.concat([df_event_edges, df_price_edges, df_lt_edges, df_of_edges], axis=0)
        elif question_type == "user query":
            # only retrieve the relations that are 1.upstream agents from the df_edges, 2.the supplier relations, and 3.the event
            df_event_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
            df_price_edges = get_price_sub_df_edges(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node)
            df_lt_edges = get_lt_sub_df_edges(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node)
            df_of_edges = get_of_sub_df_edges(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node)
            df_demand_edges = get_demand_sub_df_edges(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node)
            df_sub_edges = pd.concat([df_event_edges, df_price_edges, df_lt_edges, df_of_edges], axis=0)
        else:
            raise ValueError(f"Unknown question type: {question_type}")
        
        df_sub_edges['src'] = df_sub_edges['src'].astype(int)
        df_sub_edges['dst'] = df_sub_edges['dst'].astype(int)
        # Remove duplicate edges in df_sub_edges
        df_sub_edges = df_sub_edges.drop_duplicates(subset=['src', 'edge_attr', 'dst']).reset_index(drop=True)

        return df_sub_nodes, df_sub_edges, node_id_name_map


    def generate_text_embedding(self, nodes: pd.DataFrame, edges: pd.DataFrame):

        model, tokenizer, device = load_model[self.index_lm_model]()
        text2embedding = load_text2embedding[self.index_lm_model]

        # In SCMAS, we dont need to load nodes and edges from path. the data will be passed as input.
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
        e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src, edges.dst])
        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))

        return data


    def preprocess(self, target_stage_idx: int, target_agent_idx: int, question_type: str):
        target_node = f"stage_{target_stage_idx}_agent_{target_agent_idx}"
        env = self.retrieve_env(self.env)

        df_nodes = self.convert_env_to_node_df(env=env, event_dict=self.event_dict)
        df_edges = self.convert_env_to_edge_df(env=env, event_dict=self.event_dict, target_stage_idx=target_stage_idx, target_agent_idx=target_agent_idx) # TODO: need to write an aggregated one. To figure out the replacement of event_dict
        df_nodes.to_csv(f"df_nodes_stage_{target_stage_idx}_agent_{target_agent_idx}.csv", index=False)
        df_edges.to_csv(f"df_edges_stage_{target_stage_idx}_agent_{target_agent_idx}.csv", index=False)
        G = build_supplier_graph(df_edges=df_edges, df_nodes=df_nodes)
        df_nodes, df_edges, node_id_name_map = self.retrieve_subgraph(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node, G=G, question_type=question_type)
        graph = self.generate_text_embedding(nodes=df_nodes, edges=df_edges)
        desc = df_nodes[['node_id', 'node_attr']].to_csv(index=False)+'\n'+df_edges[['src', 'edge_attr', 'dst']].to_csv(index=False)
        with open(f"desc_stage_{target_stage_idx}_agent_{target_agent_idx}.txt", "w") as file:
            file.write(desc)

        return desc, graph, node_id_name_map
    

    def get_action_desc(self, target_stage_idx: int, target_agent_idx: int):

        env = self.retrieve_env(self.env)
        t = env['t']
        orders = env['orders'][target_stage_idx][target_agent_idx]
        num_orders = sum(orders[:, t])
        supp_idx = np.argmax(orders[:, t])
        if target_stage_idx < self.num_stages-1:
            supp_name = f"stage_{target_stage_idx+1}_agent_{supp_idx}"
        else:
            supp_name =  ""
        action_desc = (f"You just made an order of {num_orders} unit products to your supplier {supp_name} at period {t}. ")

        return action_desc
    
    def get_chat_history(self, target_stage_idx: int, target_agent_idx: int):
        chat_history = []
        try:
            with open('llama_mas_user_chat_history.csv', 'r') as file:
                reader = csv.DictReader(file)
            for row in reader:
                if int(row['stage_idx']) == target_stage_idx and int(row['agent_idx']) == target_agent_idx:
                    chat_history.append(f"User: {row['message']}")
                    chat_history.append(f"Assistant: {row['answer']}")
        except FileNotFoundError:
            print("Chat history file not found.")

        return "\n".join(chat_history)
    

    def __getitem__(self, stage_idx: int, agent_idx: int, question_type: str, message: str=None):

        target_node = f"stage_{stage_idx}_agent_{agent_idx}"
        t = self.env.period
        graph_desc, graph, node_id_name_map = self.preprocess(target_stage_idx=stage_idx, target_agent_idx=agent_idx, question_type=question_type)
        action_desc = self.get_action_desc(target_stage_idx=stage_idx, target_agent_idx=agent_idx)
        if question_type == "order placement":
            prompt = (f"You are {target_node} in the supply chain at round {t}. Based on the provided supply chain graph, answer the following question:\n\n"
                    f"Question: Considering the inventory level and the requested order, how many orders would you like to place to your supplier in this round? Answer in the form of a number. "
                    )
        elif question_type == "supplier selection":
            prompt = (f"You are {target_node} in the supply chain at round {t}. Based on the provided supply chain graph, answer the following question:\n\n"
                    f"Question: Considering the upstream agents with low price, short lead time or high order fulfillment, who you would choose as your supplier in the next round? Answer the node id of your choice."
                    # f"which of the upstream agents at stage {stage_idx+1} has the shortest lead time? Answer with the node id, e.g. 15."
                    )
        elif question_type == "user query":
            message = name2stage_agent_id(message)
            prompt = (f"User: You are {target_node} in the supply chain at round {t}. {action_desc}"
                      "Given the current state of the supply chain and your past decisions, answer the following question:\n\n"
                      f"Question: {message}"
                    )
        # print(desc)
        self.index += 1
        return {
            'id': self.index-1,
            'label': None,
            'stage_idx': stage_idx,
            'agent_idx': agent_idx,
            'desc': graph_desc,
            'graph': graph,
            'question': prompt,
            'question_type': question_type,
            "node_id_name_map": node_id_name_map,
        }

    


if __name__ == '__main__':


    dataset = SupplyChainMASDataset()
    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
