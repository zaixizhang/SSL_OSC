import argparse
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from ssl_dataset import SSLDataset, ssl_collect_fn
import numpy as np
from utils import batched_ref, delete_delta_edges, Logger, mmd_XY, mmd_linear
import torch
from sch import SchNetModel, SSLSchNetModel

paras = {
    'lr' : 1e-4,
    'epochs' : 1000,
    'cutoff' : 3,
    'property_id' : 'U0',
    'hartree_flag' : False,
    'ref_flag' : False,
    'batch_size' : 20,
    'width' : 0.1,
    'n_layers' : 2,
    'model' : 'sch',
    'diff_rate' : 0.2,
    'dataset' : 'tl_t6_train_ssl',#####
    'logger_flag' : False,#####
    'logger_name' : 't6_train_ssl', #####
    'save_flag' : True, #####
    'save_name' : 't6_train_ssl',#####
    'save_clip_epoch' : 50 #####
}


def test_error(test_loader, model, device=th.device('cuda:2')):
    MAE_fn = nn.L1Loss()

    sum = 0
    for idx, batch in enumerate(test_loader):
        X = batch.graph.to(device)
        y = batch.label.to(device)

        res, _ = model(X)
        
        mae_loss = MAE_fn(res, y)
        sum += mae_loss.detach().item()
    return sum / (idx+1)


def train(modelname="sch", epochs=80, device=th.device("cuda:2")):
    logger = Logger(paras, paras['logger_flag'], filename=paras['logger_name'])
    print(device)
    print("starting")

    ssl_dataset = SSLDataset(mode='train', dataset=paras['dataset'])
    ssl_dataloader  =  DataLoader(dataset=ssl_dataset,
                                batch_size=paras['batch_size'],
                                collate_fn=ssl_collect_fn,
                                shuffle=True,
                                num_workers=0)
    
    model = SSLSchNetModel(cutoff=paras['cutoff'], width=paras['width'], n_conv=paras['n_layers'], mmd_flag=True)

    model.to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=paras['lr'])

    print('start training!')
    for epoch in range(epochs):
        if epoch == paras['save_clip_epoch'] and paras['save_flag']:
            torch.save(model.state_dict(), './model_save/'+paras['save_name']) #####################
            print('the model has saved')

        cross_loss = 0
        node_loss = 0
        edge_loss = 0
        model.train()

        for i, batch in enumerate(ssl_dataloader):
            graph = batch.graph.to(device)
            label = batch.label.to(device)
            node_index = batch.node_index.to(device)
            one_hot = batch.one_hot.to(device)
            source_index = batch.source_index.to(device)
            target_index = batch.target_index.to(device)
            discrete_distance = batch.discrete_distance.to(device)
            select_edge_index = batch.select_edge_index.to(device)


            graph.requires_grad = False
            label.requires_grad = False
            node_index.requires_grad = False
            one_hot.requires_grad = False
            source_index.requires_grad = False
            train.requires_grad = False
            discrete_distance.requires_grad = False
            select_edge_index.requires_grad = False

            node_type, edge_type = model(graph, node_index, source_index, target_index, select_edge_index)
            one_hot = one_hot.view(-1)
            discrete_distance = discrete_distance.view(-1)
            cross_entropy_node_loss = loss_fn(node_type, one_hot.long())
            cross_entropy_edge_loss = loss_fn(edge_type, discrete_distance.long())
            total_loss = cross_entropy_edge_loss + cross_entropy_node_loss
            #print('ok')

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            cross_loss += total_loss.detach().item()
            node_loss += cross_entropy_node_loss.detach().item()
            edge_loss += cross_entropy_edge_loss.detach().item()
        cross_loss /= i + 1
        node_loss /= i + 1
        edge_loss /= i + 1

        logger.append_record("Epoch {:2d}, loss: {:.7f}".format(epoch, cross_loss))
        print("Epoch {:2d}, loss: {:.7f}, node_loss: {:.7f}, edge_loss: {:.7f}".format(
            epoch, cross_loss, node_loss, edge_loss))

        # the test error
        # if epoch % 10 == 0:
        #     test_e = test_error(test_loader, model, device)
        #     logger.append_record("----------Epoch {:2d}, test_mae: {:.7f}----------".format(epoch, test_e))
        #     print("----------Epoch {:2d}, test_mae: {:.7f}----------".format(epoch, test_e))
            

if __name__ == "__main__":
    device = th.device('cuda:2' if th.cuda.is_available() else 'cpu')
    train(paras['model'], paras['epochs'], device)