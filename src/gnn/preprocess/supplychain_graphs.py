import re
import os
import torch
import pandas as pd
import sys

from tqdm import tqdm
from torch_geometric.data.data import Data
import sys
# sys.path.append('/data/yanjia/MAS_SupplyChain')
from generate_split import generate_split
from lm_modeling import load_model, load_text2embedding

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='graph_4_4')

args = parser.parse_args()
model_name = 'sbert'
path = f'src/gnn/gnn_dataset/{args.dataset}'


# Set to use only one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_text_embedding(path: str):

    def _encode_questions(df: pd.DataFrame, filename: str):
        q_embs = text2embedding(model, tokenizer, device, df.question.tolist())
        torch.save(q_embs, f'{path}/q_{filename}_embs.pt')

    def _encode_graph():
        print('Encoding graphs...')
        os.makedirs(f'{path}/graphs', exist_ok=True)
        for i in tqdm(data_list):
            nodes = pd.read_csv(f'{path}/nodes/{i}.csv')
            edges = pd.read_csv(f'{path}/edges/{i}.csv')
            x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.LongTensor([edges.src, edges.dst])
            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
            torch.save(data, f'{path}/graphs/{i}.pt')


    # _encode_graph()

    # df_train = pd.read_csv(f'{path}/all_train_questions.csv')
    # df_test = pd.read_csv(f'{path}/all_test_questions.csv')
    os.makedirs(f'{path}/graphs/', exist_ok=True)
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # _encode_questions(df=df_train, filename='train')
    # _encode_questions(df=df_test, filename='test')
    _encode_graph()



if __name__ == '__main__':

    for_train = True
    if for_train:
        path = os.path.join(path, 'train_data')
    else:
        path = os.path.join(path, 'test_data')

    data_list = [data.strip('.csv') for data in os.listdir(f'{path}/nodes') if data.endswith('.csv')]
    num_data = len(data_list)

    print("#data in total: ", num_data)
    df_events = pd.read_csv(f'{path}/all_event_questions.csv')
    num_events_qa = len(df_events)
    df_order_fulfill_qa = pd.read_csv(f'{path}/all_order_fulfill_questions.csv')
    num_order_fulfill_qa = len(df_order_fulfill_qa)
    df_price = pd.read_csv(f'{path}/all_price_questions.csv')
    num_price_qa = len(df_price)
    df_lead_time = pd.read_csv(f'{path}/all_lead_time_questions.csv')
    num_lead_time_qa = len(df_lead_time)
    df_demand_qa = pd.read_csv(f'{path}/all_demand_questions.csv')
    num_demand_qa = len(df_demand_qa)

    generate_text_embedding(path=path)

    if for_train:

        generate_split(num_events_qa, f'{path}/split/events_qa')
        generate_split(num_order_fulfill_qa, f'{path}/split/order_fulfill_qa')
        generate_split(num_price_qa, f'{path}/split/price_qa')
        generate_split(num_lead_time_qa, f'{path}/split/lead_time_qa')
        generate_split(num_demand_qa, f'{path}/split/demand_qa')

    print("Done!")