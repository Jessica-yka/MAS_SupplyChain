## write a GCN based model for graph representation learning
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import time
import numpy as np
from sklearn.model_selection import train_test_split
import random
from dgl.data import CoraGraphDataset


gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(8, 4)
        self.layer2 = GCNLayer(4, 1)
        # self.layer1 = GCNLayer(1433, 16)
        # self.layer2 = GCNLayer(16, 7)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x
    
net = Net()


def load_sc_data(agent_profiles, num_stage: int=4):
    
    node_feat_dict = {}
    edge_feat_dict = {}
    pos_labels = [] # (sup, dem)
    neg_labels = [] # (sup, dem)

    for ap in agent_profiles:
        sup_agent_idx = f"stage_{stage}_agent_{agent}"
        node_feat_dict[sup_agent_idx] = ap.get_node_features()
        stage = ap.stage
        agent = ap.agent
        if stage < num_stage - 1:            
            for dem, label in enumerate(ap.suppliers):
                dem_agent_idx = f"stage_{stage-1}_agent_{dem}"
                if label == 1:
                    pos_labels.append((sup_agent_idx, dem_agent_idx))
                else: # label is 0 
                    neg_labels.append((sup_agent_idx, dem_agent_idx))

    labels = pos_labels + neg_labels
    return node_feat_dict, edge_feat_dict, labels



class LinkPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(LinkPredictor, self).__init__()
        self.layer1 = GCNLayer(in_feats, hidden_feats)
        self.layer2 = GCNLayer(hidden_feats, out_feats)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x

def compute_loss(pos_score, neg_score):
    scores = th.cat([pos_score, neg_score])
    labels = th.cat([th.ones(pos_score.shape[0]), th.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_scores(g, model, edges):
    with th.no_grad():
        g.ndata['h'] = model(g, g.ndata['feat'])
        g.apply_edges(fn.u_dot_v('h', 'h', 'score'), edges)
        return g.edata['score']

def train_link_predictor(g, features, pos_edges, neg_edges, epochs=50, lr=1e-2):
    model = LinkPredictor(features.shape[1], 16, 1)
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        pos_score = compute_scores(g, model, pos_edges)
        neg_score = compute_scores(g, model, neg_edges)
        loss = compute_loss(pos_score, neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:05d} | Loss {loss.item():.4f}")

# Example usage with Cora dataset
g, features, labels, train_mask, test_mask = load_sc_data()
u, v = g.edges()
eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
train_pos_u, train_pos_v = u[eids[:train_size]], v[eids[:train_size]]
test_pos_u, test_pos_v = u[eids[train_size:]], v[eids[train_size:]]

train_neg_u = np.random.choice(g.number_of_nodes(), train_size)
train_neg_v = np.random.choice(g.number_of_nodes(), train_size)
test_neg_u = np.random.choice(g.number_of_nodes(), test_size)
test_neg_v = np.random.choice(g.number_of_nodes(), test_size)

train_pos_edges = (train_pos_u, train_pos_v)
train_neg_edges = (train_neg_u, train_neg_v)
test_pos_edges = (test_pos_u, test_pos_v)
test_neg_edges = (test_neg_u, test_neg_v)

train_link_predictor(g, features, train_pos_edges, train_neg_edges)