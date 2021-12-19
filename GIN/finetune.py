import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

criterion = nn.MSELoss(reduce=True)

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        #is_valid = y**2 > 0
        #Loss matrix
        #loss_mat = criterion(pred.double(), y)
        #loss matrix after removing null target
        #loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = criterion(pred.double(), y)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.extend(batch.y.view(pred.shape).cpu().numpy())
        y_scores.extend(pred.cpu().numpy())

    y_scores = np.array(y_scores)
    y_true = np.array(y_true)

    return np.sum(abs(y_true-y_scores))/len(y_true), np.sum((y_true-y_scores)**2)/len(y_true)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'organic', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = './model_gin/mask40.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type=str, default='./model_gin/finetune',
                        help='filename to output the model')
    parser.add_argument('--filename', type=str, default = 'opv', help='output filename')
    parser.add_argument('--seed', type=int, default=41, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=3, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="already", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == "organic":
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    #dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    #print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    elif args.split == "already":
        dataset = MoleculeDataset("dataset/organic_train", dataset="organic")
        train_dataset, _, _ = random_split(dataset, null_value=0, frac_train=0.0125, frac_valid=0.8,
                                                                  frac_test=0.1875, seed=args.seed)
        valid_dataset = MoleculeDataset("dataset/organic_val", dataset="organic")
        test_dataset = MoleculeDataset("dataset/organic_test", dataset="organic")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_mae, train_mse = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_mae = 0
        val_mae, val_mse = eval(args, model, device, val_loader)
        test_mae, test_mse = eval(args, model, device, test_loader)

        print("train_mae: %f val_mae: %f test_mae: %f" %(train_mae, val_mae, test_mae))
        print("train_mse: %f val_mse: %f test_mse: %f" %(train_mse, val_mse, test_mse))

        val_acc_list.append(val_mae)
        test_acc_list.append(test_mae)
        train_acc_list.append(train_mae)

        print("")
    print('best mae: ', test_acc_list[val_acc_list.index(min(val_acc_list))])

    np.save('./training_history/masktrain_'+args.dataset+'.npy', np.array(train_acc_list))
    np.save('./training_history/maskval_'+args.dataset+'.npy', np.array(val_acc_list))
    np.save('./training_history/masktest_' + args.dataset + '.npy', np.array(test_acc_list))

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")

if __name__ == "__main__":
    main()
