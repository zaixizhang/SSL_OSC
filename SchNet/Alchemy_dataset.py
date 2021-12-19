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
_urls = {'Alchemy': 'https://alchemy.tencent.com/data/'}


class AlchemyBatcher:
    def __init__(self, graph=None, label=None):
        self.graph = graph
        self.label = label

def batcher():
    def batcher_dev(batch):
        graphs, labels = zip(*batch)
        batch_graphs = dgl.batch(graphs)
        labels = torch.stack(labels, 0)
        return AlchemyBatcher(graph=batch_graphs, label=labels)

    return batcher_dev

class DomainBatcher:
    def __init__(self, graph=None, label=None, domain_type=None):
        self.graph = graph
        self.label = label
        self.domain_type = domain_type

def batcher_domain():
    def batcher_dev(batch):
        graphs, label, domain_type = zip(*batch)
        batch_graphs = dgl.batch(graphs)
        labels = torch.stack(label, 0)
        domain_types = torch.stack(domain_type, 0)
        return DomainBatcher(graph=batch_graphs, labels=labels, domain_types=domain_types)

    return batcher_dev


class TencentAlchemyDataset(Dataset):

        # t2 = 0
        # t6 = 1
    def __init__(self, mode='train', transform=None, dataset='qm9', domain_type=0, domain_flag=False):
        # download(_urls['Alchemy'] + "%s.zip" % mode,
        #          path=str(self.zip_file_path))
        # if not os.path.exists(str(self.file_dir)):
        #     archive = zipfile.ZipFile(self.zip_file_path)
        #     archive.extractall('./Alchemy_data')
        #     archive.close()
        self.graphs, self.labels = [], []
        self.dataset = dataset
        self.domain_type = domain_type
        self.domain_flag = domain_flag

        if self.dataset == 'qm9':
            if mode == 'train':
                self.data_path = qm9_train_data_path
                self.label_path = qm9_train_label_path
            elif mode == 'test':
                self.data_path = qm9_test_data_path
                self.label_path = qm9_test_label_path
            elif mode == 'valid':
                self.data_path = qm9_valid_data_path
                self.label_path = qm9_valid_label_path
        elif dataset == 'tx':
            if mode == 'train':
                self.data_path = tx_train_data_path
            elif mode == 'test':
                self.data_path = tx_test_data_path
            elif mode == 'valid':
                self.data_path = tx_valid_data_path
        elif dataset == 'tx_t3':
            self.data_path = './dataset/tx_json/t3_mix.pkl'
        elif dataset == 'tl_t2':
            self.data_path = './dataset/dataset_for_tl/t2/t2.pkl'
        elif dataset == 'tl_t3':
            self.data_path = './dataset/dataset_for_tl/t3/t3.pkl'
        elif dataset == 'tl_t3_semi':
            self.data_path = './dataset/dataset_for_tl/t3/t3_labeled.pkl'
        elif dataset == 'tl_t6':
            self.data_path = './dataset/dataset_for_tl/t6/t6.pkl'
        elif dataset == 'tl_t6_labeled':
            self.data_path = './dataset/dataset_for_tl/t6/t6_dipole_500.pkl'
        elif dataset == 'tl_t6_test':
            self.data_path = './dataset/dataset_for_tl/t6/t6_dipole_test.pkl'
        elif dataset == 'tl_t10_labeled':
            self.data_path = './dataset/dataset_for_tl/t10/t10_labeled.pkl'
        elif dataset == 'tl_t10_test':
            self.data_path = './dataset/dataset_for_tl/t10/t10_test.pkl'
        elif dataset == 'tl_t6_train':
            self.data_path = './dataset/dataset_for_tl/t6/t6_train.pkl'
        elif dataset == 'tl_t6_pseudo':
            self.data_path = './dataset/dataset_for_tl/t6/t6_train_pseudo.pkl'
        elif dataset == 'tl_t2_4w':
            self.data_path = './dataset/dataset_for_tl/t2/t2_4w.pkl'

        self._load()

    def _load(self):
        print('-------------------------------------')
        if self.dataset == 'qm9':
            print('1')
            print('loading the dataset and label...')
            self.graphs = pickle.load(open(self.data_path,'rb'))
            label_temp = pickle.load(open(self.label_path, 'rb'))
            self.labels = torch.FloatTensor(label_temp).view(-1,1)
        elif self.dataset == 'tx':
            print('2')
            print('loading the dataset and label...')
            data = list(pickle.load(open(self.data_path, 'rb')))
            self.graphs = [data[i][0] for i in range(len(data))]
            label_temp = [data[i][1][0] for i in range(len(data))]
            self.labels = torch.FloatTensor(label_temp).view(-1,1)
        elif self.dataset == 'tx_t3':
            print('3')
            print('loading the dataset and label...')
            data = pickle.load(open(self.data_path, 'rb'))
            self.graphs = [data[i][0] for i in range(len(data))]
            label_temp = [data[i][1][0] for i in range(len(data))]
            self.labels = torch.FloatTensor(label_temp).view(-1,1)
        elif self.dataset == 'tl_t2' or self.dataset == 'tl_t3':
            print('5')
            print('loading the dataset and label for tl dataset ' + self.dataset)
            data = pickle.load(open(self.data_path, 'rb'))
            self.graphs = data[0]
            self.labels = torch.FloatTensor(data[1]).view(-1,1)
        elif self.dataset == 'tl_t3_semi':
            print('6')
            print('loading the dataset and label for tl_t3_semi')
            data = pickle.load(open(self.data_path, 'rb'))
            self.graphs = data[0][0]
            self.labels = torch.FloatTensor(data[0][1]).view(-1,1)
        elif self.dataset == 'tl_t6' or self.dataset == 'tl_t6_labeled' or self.dataset == 'tl_t6_test':
            print('7')
            print('loading the dataset and label for tl dataset' + self.dataset)
            data = pickle.load(open(self.data_path, 'rb'))
            self.graphs = data[0]
            self.labels = torch.FloatTensor(data[1]).view(-1,1)
        elif self.dataset == 'tl_t10_labeled' or self.dataset == 'tl_t10_test':
            print('8')
            print('loading the dataset and label for tl dataset ' + self.dataset)
            data = pickle.load(open(self.data_path, 'rb'))
            self.graphs = data[0]
            self.labels = torch.FloatTensor(data[1]).view(-1,1)
        elif self.dataset == 'tl_t6_train':
            print('9')
            print('loading the dataset and label for tl dataset ' + self.dataset)
            data = pickle.load(open(self.data_path, 'rb'))
            self.graphs = data[0]
            self.labels = torch.FloatTensor(data[1]).view(-1,1)
        elif self.dataset == 'tl_t6_pseudo':
            print('10')
            print('loading the dataset and label for tl dataset ' + self.dataset)
            data = pickle.load(open(self.data_path, 'rb'))
            pseudo_path = './dataset/dataset_for_tl/t6/t6_train_pseudo.pkl'
            pseudo_lable = pickle.load(open(pseudo_path, 'rb'))
            self.graphs = data[0]
            self.labels = torch.FloatTensor(pseudo_lable).view(-1,1)
        elif self.dataset == 'tl_t2_4w':
            print('11')
            print('loading the dataset and label for tl dataset' + self.dataset)
            data = pickle.load(open(self.data_path, 'rb'))
            self.graphs = data[0]
            self.labels = torch.FloatTensor(data[1]).view(-1,1)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g, l = self.graphs[idx], self.labels[idx]
        #if self.transform:
        #g = self.transform(g)
        if not self.domain_flag:
            return g, l
        else:
            return g, l, torch.Tensor([self.domain_type])


if __name__ == '__main__':
    print('starting')
    alchemy_dataset = TencentAlchemyDataset(mode='valid', dataset='tx', domain_flag=True)
    print('ok---------------------------------------------------------------')
    #print(alchemy_dataset.file_dir)
    device = torch.device('cpu')
    # To speed up the training with multi-process data loader,
    # the num_workers could be set to > 1 to
    alchemy_loader = DataLoader(dataset=alchemy_dataset,
                                batch_size=20,
                                collate_fn=batcher_domain(),
                                shuffle=False,
                                num_workers=0)

    for step, batch in enumerate(alchemy_loader):
        print("bs =", batch.graph.batch_size)
        print('edge distance size =', batch.graph.edata['distance'].size())
        break