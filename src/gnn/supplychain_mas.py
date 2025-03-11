import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('/data/yanjia/MAS_SupplyChain')
# from src.gnn.preprocess.utils.retrieval import retrieval_via_pcst
import os
from tqdm import tqdm

PATH = 'src/gnn/gnn_dataset/large_graph_test/test_data'

path_nodes = f'{PATH}/nodes'
path_edges = f'{PATH}/edges'
path_graphs = f'{PATH}/graphs'

cached_graph = f'{PATH}/cached_graphs'
cached_desc = f'{PATH}/cached_desc'


class SupplyChainMASDataset(Dataset):
    def __init__(self, event_dataset: str=None, supplier_dataset: str=None, price_dataset: str=None, lead_time_dataset: str=None):
        super().__init__()

        self.use_event = False
        self.use_supplier = False
        self.use_price = False
        self.use_lead_time = False
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

    def get_downstream_question(self, id, price_out=None, lead_time_out=None, event_put=None, supplier_out=None):
        
        # price_agent = price_out['pred']
        # lead_time_agent = lead_time_out['pred']
        # prompt = (f"Context: you think {price_agent} is the upstream agent with the lowest price. You think {lead_time_agent} is the upstream agent with the shortest lead time. Given this information, answer the following questions:\n\n"
        #           f"Question: Who you would choose as your supplier in the next round? Please consider the price and lead time of the upstream agents. Answer the node id of your choice'.\n\n")
        
        intro = price_out['question'][0].split('. ')[0]
        prompt = (intro + '. '
                  f"Question: Considering the price and lead time of the upstream agents, who you would choose as your supplier in the next round? Answer the node id of your choice and provide some reasons.\n\n")
        label = None
        data_index = lead_time_out['id'][0]
        graph = torch.load(f'{cached_graph}/{data_index}.pt')
        desc = open(f'{cached_desc}/{data_index}.txt', 'r').read()
        print(prompt)
        return {
            'id': id,
            "price_id": price_out['id'],
            "lead_time_id": lead_time_out['id'],
            'label': label,
            'desc': desc,
            'graph': graph,
            'question': prompt
        }

def preprocess(filename: str, require_retrieve: bool):

    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)

    questions = pd.read_csv(f'{PATH}/all_{filename}_questions.csv')
    if require_retrieve:
        q_embs = torch.load(f'{PATH}/q_{filename}_embs.pt')
    for index in tqdm(range(len(questions))):
        data_idx = questions.iloc[index]['graph_idx']
        # if os.path.exists(f'{cached_graph}/{graph_idx}.pt'
        #     continue
        graph = torch.load(f'{path_graphs}/{data_idx}.pt')
        nodes = pd.read_csv(f'{path_nodes}/{data_idx}.csv')
        edges = pd.read_csv(f'{path_edges}/{data_idx}.csv')
        if require_retrieve:
            subg, desc = retrieval_via_pcst(graph, q_embs[data_idx], nodes, edges, topk=5, topk_e=5, cost_e=0.5)
        else:
            subg = graph
            desc = nodes[['id', 'node_attr']].to_csv(index=False)+'\n'+edges[['src', 'edge_attr', 'dst']].to_csv(index=False)

        torch.save(subg, f'{cached_graph}/{data_idx}.pt')
        open(f'{cached_desc}/{data_idx}.txt', 'w').write(desc)


if __name__ == '__main__':

    # preprocess(filename='event', require_retrieve=False)
    # preprocess(filename='supplier', require_retrieve=False)
    preprocess(filename='price', require_retrieve=False)
    preprocess(filename='lead_time', require_retrieve=False)

    dataset = SupplyChainMASDataset(path=PATH, price_dataset='all_price_questions.csv', lead_time_dataset='all_lead_time_questions.csv')
    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
