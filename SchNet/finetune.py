import pickle
import dgl
import torch
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from Alchemy_dataset import TencentAlchemyDataset, batcher, batcher_domain
import numpy as np
from utils import batched_ref, delete_delta_edges, Logger
from model import *
from sch import SchNetModel

paras = {
    'lr' : 1e-4,
    'epochs' : 1000,
    'cutoff' : 5.0,
    'property_id' : 'U0',
    'hartree_flag' : False,
    'ref_flag' : False,
    'batch_size' : 64,
    'width' : 1,
    'delta_flag' : False,
    'delta' : 3,
    'n_layers' : 3,
    'model' : 'sch',
    'diff_rate' : 0.2,
    'logger_flag' : True,#####
    'logger_name' : 't6_finetune', #####
    'save_flag' : False, #####
    'save_name' : 't6_finetune',#####
    'save_clip_epoch' : 150, #####
}

t6_ssl_model_path = '/home/jeffzhu/MultiLayer/model_save/t6_train_ssl'

f_paras = {
    'checkpoint_path': t6_ssl_model_path,
    'finetune_dataset': 'tl_t6_labeled',
    'test_dataset': 'tl_t6_test',
    'test_epochs': 10
}

paras['epochs'] = 500
paras['lr'] = 1e-4


def test_error(test_loader, model, device=th.device('cuda:3')):
    MAE_fn = nn.L1Loss()
    model.eval()
    sum = 0
    for idx, batch in enumerate(test_loader):
        X = batch.graph.to(device)
        y = batch.label.to(device)

        res = model(X)

        mae_loss = MAE_fn(res, y)
        sum += mae_loss.detach().item()
    return sum / (idx + 1)


def train(modelname="sch", epochs=200, device=th.device("cuda:3")):
    # the setting of the dataset
    logger = Logger(paras, paras['logger_flag'], filename=paras['logger_name'])
    print(device)
    print("starting")
    alchemy_dataset = TencentAlchemyDataset(mode='train', dataset=f_paras['finetune_dataset'])
    alchemy_loader = DataLoader(dataset=alchemy_dataset,
                                batch_size=paras['batch_size'],
                                collate_fn=batcher(),
                                shuffle=True)

    # the setting of model structure
    model = SchNetModel(cutoff=paras['cutoff'], width=paras['width'], n_conv=paras['n_layers'])

    # the setting of checkpoint
    '''
    checkpoint = torch.load(f_paras['checkpoint_path'])
    model.load_state_dict(checkpoint, strict=False)
    '''
    model.to(device)
    print(model)

    # the setting of test dataset
    test_dataset = TencentAlchemyDataset(mode='train', dataset=f_paras['test_dataset'])
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=paras['batch_size'],
                             collate_fn=batcher(),
                             shuffle=True,
                             num_workers=0)

    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    optimizer = th.optim.Adam(model.parameters(), lr=paras['lr'])

    print('start training!')
    for epoch in range(epochs):
        if epoch == paras['save_clip_epoch'] and paras['save_flag']:
            torch.save(model.state_dict(),
                       './model_save/' + paras['save_name'])  #####################
            print('the model has saved')

        w_loss, w_mae = 0, 0
        model.train()

        for idx, batch in enumerate(alchemy_loader):
            X = batch.graph.to(device)
            y = batch.label.to(device)
            # print(X.edata['distance'].size())

            if paras['delta_flag'] == True:
                temp = np.array(torch.squeeze(X.edata['distance']).tolist())
                index_list = np.where(temp > paras['delta'])[0]
                X.remove_edges(index_list)

            if paras['ref_flag']:
                y_ref = batched_ref(X.ndata['node_type'].tolist(), X.batch_num_nodes, paras['hartree_flag'],
                                    paras['property_id'])
                y_ref = torch.FloatTensor(y_ref).to(device)
                y = y - y_ref

            X.requires_grad = False
            y.requires_grad = False

            if modelname == 'test':
                res = model(X, 3)
            else:
                res = model(X)
            loss = loss_fn(res, y)
            mae = MAE_fn(res, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            w_mae += mae.detach().item()
            w_loss += loss.detach().item()
        # the test error
        if epoch % f_paras['test_epochs'] == 0:
            test_e = test_error(test_loader, model, device)
            logger.append_record("----------Epoch {:2d}, test_mae: {:.7f}----------".format(epoch, test_e))
            print("----------Epoch {:2d}, test_mae: {:.7f}----------".format(epoch, test_e))
        w_mae /= idx + 1

        logger.append_record("Epoch {:2d}, loss: {:.7f}, mae: {:.7f}".format(epoch, w_loss, w_mae))
        print("Epoch {:2d}, loss: {:.7f}, mae: {:.7f}".format(epoch, w_loss, w_mae))


if __name__ == "__main__":
    train(epochs=paras['epochs'])