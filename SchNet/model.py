import dgl
import torch
import torch as th
import torch.nn as nn
from layers import AtomEmbedding, Interaction, ShiftSoftplus, RBFLayer, CFConvCluster, InteractionCluster
import copy
import dgl
from utils import *
import torch.nn.functional as F
import numpy as np
import pickle


class TestModel(nn.Module):
    def __init__(self, 
                dim=64,
                low=0,
                cutoff=5,
                width=0.1,
                out_dim = 1,
                n_conv=3,
                diff_rate=0.2):
        super().__init__()
        print('it is test_model ... ...')
        self.dim = dim
        self.low = low
        self.cutoff= cutoff
        self.width = width
        self.out_dim = out_dim
        self.n_conv = n_conv
        self.activation = ShiftSoftplus()
        self.diff_rate = diff_rate

        self.embedding_layer = AtomEmbedding(dim=self.dim, type_num=100)
        self.rbf_layer = RBFLayer(low=self.low, high=self.cutoff, gap=self.width)
        self.node_layer1 = nn.Linear(dim, dim, bias=False)
        
        self.interaction_layers = nn.ModuleList([InteractionCluster(self.rbf_layer._fan_out, self.dim, self.diff_rate) for i in range(self.n_conv)])

        self.atom_dense_layer1 = nn.Linear(self.dim, 64)
        self.atom_dense_layer2 = nn.Linear(64, self.out_dim)

    def forward(self, g, num_cluster):
        self.embedding_layer(g) # get node feature
        self.rbf_layer(g)       # get edge feature
        
        for idx in range(self.n_conv):
            self.interaction_layers[idx](g, num_cluster)
        
        atom = self.atom_dense_layer1(g.ndata["node"])
        atom = self.activation(atom)
        res = self.atom_dense_layer2(atom)
        g.ndata["res"] = res

        res = dgl.mean_nodes(g, "res")
        return res

        print('ok')
    

def filter_edge_cluster(edges):
    return (edges.src['cluster_id'] == 0) * (edges.dst['cluster_id'] == 0)


def filter_func(cluster_id):
    def filter_edge_cluster(edges):
        return (edges.src['cluster_id'] == cluster_id) * (edges.dst['cluster_id'] == cluster_id)
    return filter_edge_cluster

def filter_func_diff(edges):
    temp =  edges.src['cluster_id'] != edges.dst['cluster_id']
    return temp
    

    

if __name__ == '__main__':
    g, label, map_i, num_cluster_i = pickle.load(open('./dataset/t3_mix_mol.pkl', 'rb'))
    g.ndata['node_type'] = g.ndata['node_type'].long()
    g.edata['distance'] = g.edata['distance'].view(-1,1)
    g.ndata['cluster_id'] = torch.Tensor(map_i)

    u = [i for i in range(23) for j in range(23)]
    v = [j for i in range(23) for j in range(23)]

    test_index = g.filter_edges(filter_func_diff).tolist()
    cluster_index = list(set(range(len(g.edges()[0]))) - set(test_index))
    print(len(cluster_index))
    print(len(test_index))
    print(g.filter_edges(filter_func(0)))

    model = TestModel()
    model(g, 3)