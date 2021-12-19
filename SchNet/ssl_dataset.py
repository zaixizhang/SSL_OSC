import os
import zipfile
import os.path as osp
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import dgl
from dgl.data.utils import download
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pathlib
import pandas as pd
import numpy as np
import pickle


class SSLBatcher():
    def __init__(self, graph, label, node_index, one_hot, source_index, target_index, discrete_distance, domain_type, select_edge_index):
        self.graph = graph
        self.label = label
        self.node_index = node_index
        self.one_hot = one_hot  
        self.source_index = source_index
        self.target_index = target_index
        self.discrete_distance = discrete_distance
        self.domain_type = domain_type
        self.select_edge_index = select_edge_index

def ssl_collect_fn(batch):
    g, l, index_arr, one_hot_arr, source_index, target_index, discrete_distance, domain_type, select_edge_index = zip(*batch)
    gs = dgl.batch(g)
    ls = torch.stack(l, 0)

    number_of_nodes = g[0].number_of_nodes()
    number_of_edges = number_of_nodes**2
    index_arr = [index_arr[i] + number_of_nodes * i for i in range(len(g))]
    node_index = torch.stack(index_arr, 0) 
    one_hot = torch.stack(one_hot_arr, 0)

    source_index = [source_index[i] + number_of_nodes *i for i in range(len(g))]
    target_index = [target_index[i] + number_of_nodes *i for i in range(len(g))]
    select_edge_index = [select_edge_index[i] + number_of_edges *i for i in range(len(g))]
    source_index = torch.stack(source_index, 0)
    target_index = torch.stack(target_index, 0)
    discrete_distance = torch.stack(discrete_distance, 0)
    select_edge_index = torch.stack(select_edge_index, 0)

    domain_type = torch.stack(domain_type, 0).squeeze()
    
    return SSLBatcher(graph=gs, label=ls, node_index=node_index, one_hot=one_hot, source_index=source_index, target_index=target_index, discrete_distance=discrete_distance, domain_type=domain_type, select_edge_index=select_edge_index)

class SSLDataset(Dataset):

    def __init__(self, mode='train', transform=None, dataset='t2_mix'):
        # download(_urls['Alchemy'] + "%s.zip" % mode,
        #          path=str(self.zip_file_path))
        # if not os.path.exists(str(self.file_dir)):
        #     archive = zipfile.ZipFile(self.zip_file_path)
        #     archive.extractall('./Alchemy_data')
        #     archive.close()
        self.graphs, self.labels = [], []
        self.sel_index = []
        self.one_hot = []
        self.source_index = []
        self.target_index = []
        self.discrete_distance = []
        self.domain_index = []

        self.domain_flag = None
        self.data_path = None

        self.dataset = dataset

        if self.dataset == 'tl_t6_test_ssl':
            self.ssl_node_data_path = './dataset/dataset_for_tl/t6/t6_test_sslnode.pkl'
            self.data_path = './dataset/dataset_for_tl/t6/t6_test.pkl'
            self.ssl_edge_data_path = './dataset/dataset_for_tl/t6/t6_test_ssledge.pkl'
            self.domain_flag = 6
        if self.dataset == 'tl_t6_train_ssl':
            self.ssl_node_data_path = './dataset/dataset_for_tl/t6/t6_train_sslnode.pkl'
            self.data_path = './dataset/dataset_for_tl/t6/t6_train1.pkl'
            self.ssl_edge_data_path = './dataset/dataset_for_tl/t6/t6_train_ssledge1.pkl'
            self.domain_flag = 6
        if self.dataset == 't2_mix':
            self.ssl_node_data_path = './dataset/dataset_for_tl/ssl_t2t6/t2_nodemix'
            self.data_path = './dataset/dataset_for_tl/ssl_t2t6/t2_mix'
            self.ssl_edge_data_path = './dataset/dataset_for_tl/ssl_t2t6/t2_edgemix'
            self.domain_flag = 2
        if self.dataset == 't6_mix':
            self.ssl_node_data_path = './dataset/dataset_for_tl/ssl_t2t6/t6_nodemix'
            self.data_path = './dataset/dataset_for_tl/ssl_t2t6/t6_mix'
            self.ssl_edge_data_path = './dataset/dataset_for_tl/ssl_t2t6/t6_edgemix'  
            self.domain_flag = 6

        self._load()

    def _load(self):

        data = pickle.load(open(self.data_path, 'rb'))
        self.graphs = data[0]
        self.labels = torch.FloatTensor(data[1]).view(-1,1)

        # for node_data
        ssl_node_data = pickle.load(open(self.ssl_node_data_path, 'rb'))
        self.sel_index = ssl_node_data[0]
        self.one_hot = ssl_node_data[1]

        #for edge_data
        ssl_edge_data = pickle.load(open(self.ssl_edge_data_path, 'rb'))
        self.source_index = ssl_edge_data[0]
        self.target_index = ssl_edge_data[1]
        self.discrete_distance = ssl_edge_data[2]
        self.select_edge_index = ssl_edge_data[3]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g, l = self.graphs[idx], self.labels[idx]
        index_arr, one_hot_arr = self.sel_index[idx], self.one_hot[idx]
        source_index, target_index = self.source_index[idx], self.target_index[idx]
        discrete_distance = self.discrete_distance[idx]
        select_edge_index = self.select_edge_index[idx]
        if self.domain_flag == 6:
            domain_type = torch.LongTensor([1])
        elif self.domain_flag == 2:
            domain_type = torch.LongTensor([0])
        #if self.transform:
        #g = self.transform(g)
        return g, l, index_arr, one_hot_arr, source_index, target_index, discrete_distance, domain_type, select_edge_index


if __name__ == '__main__':
    ssl_dataset = SSLDataset(dataset='t6_mix')
    ssl_loader = DataLoader(dataset=ssl_dataset,
                            batch_size=20,
                            collate_fn=ssl_collect_fn,
                            shuffle=False,
                            num_workers=0)

    for step, batch in enumerate(ssl_loader):
        print("bs =", batch.domain_type.size())
        print('edge distance size =', batch.graph.edata['distance'].size())
