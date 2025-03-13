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
from utils.retrieval import retrieve_subgraph, get_sub_df_nodes
from utils.utils import load_json_to_graph
import json


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='graph_4_4')
parser.add_argument('--for_train', action='store_true')
parser.add_argument('--require_retrieval', action='store_true')

args = parser.parse_args()
model_name = 'sbert'
PATH = f'src/gnn/gnn_dataset/{args.dataset}'



# Set to use only one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_text_embedding(path: str, require_retrieve: bool = True):

    # def _encode_questions(df: pd.DataFrame, filename: str):
    #     q_embs = text2embedding(model, tokenizer, device, df.question.tolist())
    #     torch.save(q_embs, f'{path}/q_{filename}_embs.pt')

    def _encode_graph():
        print('Encoding graphs...')
        os.makedirs(f'{path}/graphs', exist_ok=True)
        for i in tqdm(data_list):
            if require_retrieve:
                nodes = pd.read_csv(f'{cached_nodes}/{i}.csv')
                edges = pd.read_csv(f'{cached_edges}/{i}.csv')
                x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
                e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
                edge_index = torch.LongTensor([edges.src, edges.dst])
                data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
                torch.save(data, f'{cached_graphs}/{i}.pt')
            else:
                nodes = pd.read_csv(f'{path_nodes}/{i}.csv')
                edges = pd.read_csv(f'{path_edges}/{i}.csv')
                x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
                e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
                edge_index = torch.LongTensor([edges.src, edges.dst])
                data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
                torch.save(data, f'{path_graphs}/{i}.pt')


    # _encode_graph()

    os.makedirs(f'{path_graphs}', exist_ok=True)
    os.makedirs(f'{cached_graphs}', exist_ok=True)
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # _encode_questions(df=df_train, filename='train')
    # _encode_questions(df=df_test, filename='test')
    _encode_graph()


def preprocess(filename: str, require_retrieval: bool):

    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_nodes, exist_ok=True)
    os.makedirs(cached_edges, exist_ok=True)

    questions = pd.read_csv(f'{path}/all_{filename}_questions.csv')
    
    for index in tqdm(range(len(questions))):
        data_idx = questions.iloc[index]['graph_idx']
        target_node = questions.iloc[index]['target_node']

        df_nodes = pd.read_csv(f'{path_nodes}/{data_idx}.csv')
        df_edges = pd.read_csv(f'{path_edges}/{data_idx}.csv')

        env = json.load(open(f'{path_env}/{data_idx}.json'))
        G = load_json_to_graph(f"{path_G}/{data_idx}.json")

        if require_retrieval:
            df_nodes, df_edges = retrieve_subgraph(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node, env=env, G=G, data_type=filename, path=f'{path_graph_imgs}/{data_idx}.png')
            df_nodes.to_csv(f'{cached_nodes}/{data_idx}.csv', index=False)
            df_edges.to_csv(f'{cached_edges}/{data_idx}.csv', index=False)

            desc = df_nodes[['node_id', 'node_attr']].to_csv(index=False)+'\n'+df_edges[['src', 'edge_attr', 'dst']].to_csv(index=False)
            # desc = df_nodes.to_csv(index=False)+'\n'+df_edges.to_csv(index=False)
            # torch.save(subg, f'{cached_graph}/{data_idx}.pt')
            open(f'{cached_desc}/{data_idx}.txt', 'w').write(desc)
        else:
            df_nodes = get_sub_df_nodes(df_nodes=df_nodes, target_node=target_node)
            desc = df_nodes[['node_id', 'node_attr']].to_csv(index=False)+'\n'+df_edges[['src', 'edge_attr', 'dst']].to_csv(index=False)
            open(f'{path_desc}/{data_idx}.txt', 'w').write(desc)



def preprocess_kaping(filename: str):

    kaping_desc = f'{PATH}/{data_type}_data/cached_kaping_desc'
    os.makedirs(kaping_desc, exist_ok=True)

    questions = pd.read_csv(f'{path}/all_{filename}_questions.csv')
    for index in tqdm(range(len(questions))):
        data_idx = questions.iloc[index]['graph_idx']
        target_node = questions.iloc[index]['target_node']

        df_nodes = pd.read_csv(f'{path_nodes}/{data_idx}.csv')
        df_edges = pd.read_csv(f'{path_edges}/{data_idx}.csv')

        # retrieve related subgraphs
        df_nodes, df_edges = retrieve_subgraph(df_nodes=df_nodes, df_edges=df_edges, target_node=target_node, data_type=filename)
        desc = []
        for i in range(len(df_nodes)):
            if len(df_nodes.iloc[i]['node_attr'].split(",")) > 0:
                desc.append(f"({df_nodes.iloc[i]})")
        for i in range(len(df_edges)):
            desc.append(f"({df_edges.iloc[i]})")

        desc = "\n".join(desc)
        open(f'{kaping_desc}/{data_idx}.txt', 'w').write(desc)



if __name__ == '__main__':

    for_train = args.for_train
    require_retrieval = args.require_retrieval

    print("Generating data for training..." if for_train else "Generating data for testing...")
    print("Do retrieval..." if require_retrieval else "Skip retrieval...")

    data_type = "train" if for_train else "test"

    path = os.path.join(PATH, f'{data_type}_data')
    path_nodes = f'{PATH}/{data_type}_data/nodes'
    path_edges = f'{PATH}/{data_type}_data/edges'
    path_graphs = f'{PATH}/{data_type}_data/graphs'
    path_graph_imgs = f'{PATH}/{data_type}_data/graph_imgs'
    path_env = f'{PATH}/{data_type}_data/envs'
    path_G = f'{PATH}/{data_type}_data/G'
    path_desc = f'{PATH}/{data_type}_data/desc'
    os.makedirs(path_desc, exist_ok=True)

    cached_graphs = f'{PATH}/{data_type}_data/cached_graphs'
    cached_desc = f'{PATH}/{data_type}_data/cached_desc'
    cached_nodes = f'{PATH}/{data_type}_data/cached_nodes'
    cached_edges = f'{PATH}/{data_type}_data/cached_edges'

    import concurrent.futures

    def parallel_preprocess(filename):
        preprocess(filename=filename, require_retrieval=require_retrieval)

    filenames = ['event', 'order_fulfill', 'price', 'lead_time', 'demand']

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(parallel_preprocess, filenames)
    # preprocess(filename='event', require_retrieval=require_retrieval)
    # preprocess(filename='order_fulfill', require_retrieval=require_retrieval)
    # preprocess(filename='price', require_retrieval=require_retrieval)
    # preprocess(filename='lead_time', require_retrieval=require_retrieval)
    # preprocess(filename='demand', require_retrieval=require_retrieval)

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