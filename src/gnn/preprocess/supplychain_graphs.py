import re
import os
import torch
import pandas as pd
import sys

from tqdm import tqdm
from torch_geometric.data.data import Data
import sys
sys.path.append('/data/yanjia/MAS_SupplyChain')
from src.gnn.preprocess.generate_split import generate_split
from src.gnn.preprocess.lm_modeling import load_model, load_text2embedding

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='large_graph_test')

args = parser.parse_args()
model_name = 'sbert'
path = f'src/gnn/gnn_dataset/{args.dataset}'
# num_graph = len([data for data in os.listdir(f'{path}/nodes') if data.endswith('.csv')])
# num_data_per_graph = 5
# num_data = num_graph * num_data_per_graph
num_data = len([data for data in os.listdir(f'{path}/nodes') if data.endswith('.csv')])
df_train = pd.read_csv(f'{path}/all_train_questions.csv')
num_train_data = len(df_train)
print("#data in total: ", num_data)
# Set to use only one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_text_embedding(path: str):

    def _encode_questions(df: pd.DataFrame, filename: str):
        q_embs = text2embedding(model, tokenizer, device, df.question.tolist())
        torch.save(q_embs, f'{path}/q_{filename}_embs.pt')

    def _encode_graph():
        print('Encoding graphs...')
        os.makedirs(f'{path}/graphs', exist_ok=True)
        for i in tqdm(range(num_data)):
            nodes = pd.read_csv(f'{path}/nodes/{i}.csv')
            edges = pd.read_csv(f'{path}/edges/{i}.csv')
            x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.LongTensor([edges.src, edges.dst])
            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
            torch.save(data, f'{path}/graphs/{i}.pt')

    # model, tokenizer, device = load_model[model_name]()
    # text2embedding = load_text2embedding[model_name]

    # _encode_graph()

    df_train = pd.read_csv(f'{path}/all_train_questions.csv')
    # df_test = pd.read_csv(f'{path}/all_test_questions.csv')
    os.makedirs(f'{path}/graphs/', exist_ok=True)
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    _encode_questions(df=df_train, filename='train')
    # _encode_questions(df=df_test, filename='test')
    _encode_graph()



if __name__ == '__main__':

    generate_text_embedding(path=path)
    generate_split(num_train_data, f'{path}/split')
    print("Done!")