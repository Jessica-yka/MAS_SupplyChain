import torch
import numpy as np
from torch_geometric.data.data import Data
import random
import networkx as nx
import pandas as pd
import os
import json


def check_connection(G, event_target_node, target_node):

    return nx.has_path(G, source=event_target_node, target=target_node)


def get_event_sub_df_edges(G: nx.DiGraph, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, target_node: str, path: str=None):

    # G_sub = G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True) if d['label']=='is the supplier of']).copy()

    related_nodes = list(nx.nodes(nx.dfs_tree(G, target_node))) + list(nx.nodes(nx.dfs_tree(G.reverse(), target_node)))
    related_nodes = dict(zip(related_nodes, [1 for _ in related_nodes]))

    node_name_id_map = dict(zip(df_nodes['name'].tolist(), df_nodes['node_id'].tolist()))
    df_simp_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
    
    row_idx = 0
    # list all the edges in G_sub and make it df_simp_edges
    # get the label of edges
    for src_name, dst_name, data in G.edges(data=True):
        if related_nodes.get(src_name, 0) and related_nodes.get(dst_name, 0):
            df_simp_edges.loc[row_idx] = [node_name_id_map[src_name], data['label'], node_name_id_map[dst_name], src_name, dst_name]
            row_idx += 1

    df_events = df_edges[df_edges['label'].str.contains('affects')].reset_index(drop=True)
    for i in range(len(df_events)):
        df_simp_edges.loc[row_idx] = [node_name_id_map[df_events.loc[i, 'source']], df_events.loc[i, 'label'], node_name_id_map[df_events.loc[i, 'target']], df_events.loc[i, 'source'], df_events.loc[i, 'target']]
        row_idx += 1

    return df_simp_edges


def get_of_sub_df_edges(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, target_node: str):

    node_name_id_map = dict(zip(df_nodes['name'].tolist(), df_nodes['node_id'].tolist()))
    df_simp_edges = df_edges[(df_edges['source']==target_node)|(df_edges['target']==target_node)].reset_index(drop=True)
    # get sub df_edges that the edge attrs contains either request or delivery
    df_simp_edges = df_simp_edges[df_simp_edges['label'].str.contains('request') | df_simp_edges['label'].str.contains('deliverying')].reset_index(drop=True)

    df_simp_edges['src'] = df_simp_edges['source'].apply(lambda x: node_name_id_map[x])
    df_simp_edges['dst'] = df_simp_edges['target'].apply(lambda x: node_name_id_map[x])
    # change the column name "label to edge_attr"
    df_simp_edges.rename(columns={'label': 'edge_attr', 'source': "src_name", "target": 'dst_name'}, inplace=True)
    # remove aspect column
    df_simp_edges.drop(columns=['aspect'], inplace=True)

    return df_simp_edges


def get_demand_sub_df_edges(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, target_node: str):

    node_name_id_map = dict(zip(df_nodes['name'].tolist(), df_nodes['node_id'].tolist()))
    df_simp_edges = df_edges[(df_edges['target']==target_node)].reset_index(drop=True)
    df_simp_edges = df_simp_edges[df_simp_edges['label'].str.contains('request')].reset_index(drop=True)

    df_simp_edges['src'] = df_simp_edges['source'].apply(lambda x: node_name_id_map[x])
    df_simp_edges['dst'] = df_simp_edges['target'].apply(lambda x: node_name_id_map[x])
    # change the column name "label to edge_attr"
    df_simp_edges.rename(columns={'label': 'edge_attr', 'source': "src_name", "target": 'dst_name'}, inplace=True)
    # remove aspect column
    df_simp_edges.drop(columns=['aspect'], inplace=True)

    return df_simp_edges


def get_lt_sub_df_edges(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, target_node: str, path: str=None):

    node_name_id_map = dict(zip(df_nodes['name'].tolist(), df_nodes['node_id'].tolist()))

    df_simp_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
    df_simp_edges = df_edges[df_edges["target"]==target_node].reset_index(drop=True)
    df_simp_edges = df_simp_edges[df_simp_edges['label'].str.contains('lead time')].reset_index(drop=True)

    df_simp_edges['src'] = df_simp_edges['source'].apply(lambda x: node_name_id_map[x])
    df_simp_edges['dst'] = df_simp_edges['target'].apply(lambda x: node_name_id_map[x])
    # change the column name "label to edge_attr"
    df_simp_edges.rename(columns={'label': 'edge_attr', 'source': "src_name", "target": 'dst_name'}, inplace=True)
    # remove aspect column
    df_simp_edges.drop(columns=['aspect'], inplace=True)

    return df_simp_edges


def get_price_sub_df_edges(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, target_node: str, path: str=None):

    node_name_id_map = dict(zip(df_nodes['name'].tolist(), df_nodes['node_id'].tolist()))
    df_simp_edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst', 'src_name', 'dst_name'])
    df_simp_edges = df_edges[df_edges["target"]==target_node].reset_index(drop=True)
    df_simp_edges = df_simp_edges[df_simp_edges['label'].str.contains('upstream agent')].reset_index(drop=True)

    df_simp_edges['src'] = df_simp_edges['source'].apply(lambda x: node_name_id_map[x])
    df_simp_edges['dst'] = df_simp_edges['target'].apply(lambda x: node_name_id_map[x])
    # change the column name "label to edge_attr"
    df_simp_edges.rename(columns={'label': 'edge_attr', 'source': "src_name", "target": 'dst_name'}, inplace=True)
    # remove aspect column
    df_simp_edges.drop(columns=['aspect'], inplace=True)

    return df_simp_edges


def get_sub_df_nodes(df_nodes: pd.DataFrame, target_node: str, path: str=None):

    df_nodes_sub = pd.DataFrame(columns=['node_id', 'node_attr', 'type', 'name'])
    # the competitors at the save stage
    for i in range(len(df_nodes)):
        # itself
        if df_nodes.loc[i, 'name'] == target_node:
            attr = (f"{df_nodes.loc[i, 'name']}: "
                    f"price: {df_nodes.loc[i, 'sale_price']}, "
                    f"production cost: {df_nodes.loc[i, 'prod_cost']}, "
                    f"production capacity: {df_nodes.loc[i, 'prod_capacity']}, "
                    f"inventory: {df_nodes.loc[i, 'inventory']}, "
                    f"backlog: {df_nodes.loc[i, 'backlog']}, "
                    f"upstream backlog: {df_nodes.loc[i, 'upstream_backlog']}")
            df_nodes_sub.loc[i, ['node_id', 'node_attr', 'type', 'name']] = [df_nodes.loc[i, 'node_id'], attr, df_nodes.loc[i, 'type'], df_nodes.loc[i, 'name']]
        # the suppliers of the target node
        elif f"stage_{df_nodes.loc[i, 'stage_id']-1}" in target_node:
            attr = (f"{df_nodes.loc[i, 'name']}: "
                    f"price: {df_nodes.loc[i, 'sale_price']}, "
                    f"production capacity: {df_nodes.loc[i, 'prod_capacity']}")
            df_nodes_sub.loc[i, ['node_id', 'node_attr', 'type', 'name']] = [df_nodes.loc[i, 'node_id'], attr, df_nodes.loc[i, 'type'], df_nodes.loc[i, 'name']]
        else: # the suppliers of the suppliers or the downstream customers
            attr = (f"{df_nodes.loc[i, 'name']}")
            df_nodes_sub.loc[i, ['node_id', 'node_attr', 'type', 'name']] = [df_nodes.loc[i, 'node_id'], attr, df_nodes.loc[i, 'type'], df_nodes.loc[i, 'name']]

    # df_nodes_sub.to_csv(path, index=False)
    return df_nodes_sub
