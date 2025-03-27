import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('/data/yanjia/MAS_SupplyChain')
# from src.gnn.preprocess.utils.retrieval import retrieval_via_pcst
import os
from tqdm import tqdm

PATH = 'src/gnn/gnn_dataset/graph_4_4/train_data'

path_nodes = f'{PATH}/nodes'
path_edges = f'{PATH}/edges'
path_graphs = f'{PATH}/graphs'

cached_graph = f'{PATH}/cached_graphs'
cached_desc = f'{PATH}/cached_desc'

class SupplyChainGraphsDataset(Dataset):
    def __init__(self, dataset='all_event_questions.csv', type: str = 'event_qa', prompting_tech=None):
        super().__init__()

        self.text = pd.read_csv(f'{PATH}/{dataset}')
        self.num_data = len(self.text)
        # self.prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.graph = None
        self.graph_type = 'Contextualized Supply Chain Graph'
        self.type = type
        self.init_prompt = "Please answer the given question."

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):

        question = self.text.loc[index, 'question']
        label = self.text.loc[index, 'label']
        data_index = int(self.text.loc[index, 'graph_idx'])

        graph = torch.load(f'{cached_graph}/{data_index}.pt')
        desc = open(f'{cached_desc}/{data_index}.txt', 'r').read()

        return {
            'id': data_index,
            'label': str(label),
            'desc': desc,
            'graph': graph,
            'question': question,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{PATH}/split/{self.type}/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/{self.type}/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/{self.type}/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}




if __name__ == '__main__':


    dataset = SupplyChainGraphsDataset(dataset='all_event_questions.csv', type='events_qa')
    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')

    dataset = SupplyChainGraphsDataset(dataset='all_order_fulfill_questions.csv', type='order_fulfill_qa')
    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')

    dataset = SupplyChainGraphsDataset(dataset='all_price_questions.csv', type='price_qa')
    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')

    dataset = SupplyChainGraphsDataset(dataset='all_lead_time_questions.csv', type='lead_time_qa')
    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')

    dataset = SupplyChainGraphsDataset(dataset='all_demand_questions.csv', type='demand_qa')
    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
