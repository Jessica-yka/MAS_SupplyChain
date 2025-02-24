import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('/home/vislab/Yanjia/MAS_SupplyChain')
from src.gnn.preprocess.utils.retrieval import retrieval_via_pcst
import os
from tqdm import tqdm

PATH = 'src/gnn/gnn_dataset/large_graph_test'
path_nodes = f'{PATH}/nodes'
path_edges = f'{PATH}/edges'
path_graphs = f'{PATH}/graphs'

cached_graph = f'{PATH}/cached_graphs'
cached_desc = f'{PATH}/cached_desc'

class SupplyChainGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.text = pd.read_csv(f'{PATH}/all_questions.csv')
        # self.prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.graph = None
        self.graph_type = 'Contextualized Supply Chain Graph'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):

        question = self.text.loc[index, 'question']
        label = self.text.loc[index, 'label']
        graph_idx = int(self.text.loc[index, 'graph_idx'])

        
        # nodes = pd.read_csv(f'{PATH}/nodes/{graph_idx}.csv')
        # edges = pd.read_csv(f'{PATH}/edges/{graph_idx}.csv')

        # graph = torch.load(f'{PATH}/graphs/{graph_idx}.pt')        
        # desc = nodes.to_csv(index=False)+'\n'+edges.to_csv(index=False)

        graph = torch.load(f'{cached_graph}/{graph_idx}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()

        return {
            'id': index,
            'label': label,
            'desc': desc,
            'graph': graph,
            'question': question,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{PATH}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


def preprocess(require_retrieve=True):

    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)

    questions = pd.read_csv(f'{PATH}/all_questions.csv')
    q_embs = torch.load(f'{PATH}/q_embs.pt')
    for index in tqdm(range(len(questions))):
        graph_idx = questions.iloc[index]['graph_idx']
        # if os.path.exists(f'{cached_graph}/{graph_idx}.pt'
        #     continue
        graph = torch.load(f'{path_graphs}/{graph_idx}.pt')
        nodes = pd.read_csv(f'{path_nodes}/{graph_idx}.csv')
        edges = pd.read_csv(f'{path_edges}/{graph_idx}.csv')
        if require_retrieve:
            subg, desc = retrieval_via_pcst(graph, q_embs[index], nodes, edges, topk=10, topk_e=10, cost_e=0.5)
        else:
            subg = graph
            desc = nodes.to_csv(index=False)+'\n'+edges.to_csv(index=False)

        torch.save(subg, f'{cached_graph}/{graph_idx}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':

    preprocess(require_retrieve=True)
    dataset = SupplyChainGraphsDataset()

    # data = dataset[0]
    # for k, v in data.items():
    #     print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
