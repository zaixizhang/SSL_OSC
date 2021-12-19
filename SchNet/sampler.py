import dgl
import numpy as np
import pickle
import torch

# hyper_parameters
MEAN = 8.4
STD = 5.2

def edge_sampler(dgl_graph, number, mean=MEAN, std=STD):
    number_of_edges = dgl_graph.number_of_edges()
    selected_edges_index = np.random.choice(number_of_edges, number, replace=False)
    selected_edges_index = sorted(selected_edges_index)

    source_index = dgl_graph.edges()[0][selected_edges_index]
    target_index = dgl_graph.edges()[1][selected_edges_index]

    distance = dgl_graph.edata['distance'][selected_edges_index]
    distance_np = distance.numpy()

    distance_np -= mean
    distance_np /= std

    standard_table = [i for i in range(-2,3)]
    res_table = [np.argmin(np.abs(standard_table - distance_np[i])) for i in range(number)]
    res_table = torch.Tensor(res_table).view(-1)
    return source_index, target_index, res_table, torch.tensor(selected_edges_index)


def node_sampler(dgl_graph, number):
    type0 = torch.LongTensor([0 for i in range(number)])
    type1 = torch.LongTensor([1 for i in range(number)])
    type2 = torch.LongTensor([2 for i in range(number)])

    num_nodes = dgl_graph.number_of_nodes()

    selected_nodes_index = np.random.choice(num_nodes, number, replace=False)
    selected_nodes_index = sorted(selected_nodes_index)

    node_type = dgl_graph.ndata['node_type'][selected_nodes_index]
    node_type = torch.where(node_type == 1, type0, node_type)
    node_type = torch.where(node_type == 6, type1, node_type)
    node_type = torch.where(node_type == 16, type2, node_type)

    return torch.Tensor(selected_nodes_index), node_type

def dataset_node_sampler(dataset_path, saved_dataset_path, number=10):
    dataset = pickle.load(open(dataset_path, 'rb'))

    graph = dataset[0]
    
    nodes_index_pool = []
    one_hot_pool = []

    cnt = 0
    for g in graph:
        print(cnt)
        cnt += 1

        selected_nodes_index, one_hot = node_sampler(g, number)
        nodes_index_pool.append(selected_nodes_index)
        one_hot_pool.append(one_hot)
    res = [nodes_index_pool, one_hot_pool]

    pickle.dump(res, open(saved_dataset_path,'wb'))

def dataset_edge_sampler(dataset_path, saved_dataset_path, number=40, mean=None, std=None):
    dataset = pickle.load(open(dataset_path, 'rb'))

    graph = dataset[0]
    
    source_index_pool = []
    target_index_pool = []
    distance_pool = []
    select_index_pool = []

    cnt = 0
    for g in graph:
        print(cnt)
        cnt += 1

        source_index, target_index, distance, selected_edges_index = edge_sampler(g, number)
        source_index_pool.append(source_index)
        target_index_pool.append(target_index)
        distance_pool.append(distance)
        select_index_pool.append(selected_edges_index)
    res = [source_index_pool, target_index_pool, distance_pool, select_index_pool]

    pickle.dump(res, open(saved_dataset_path,'wb'))


if __name__ == "__main__":
    dataset_path = './dataset/dataset_for_tl/t6/t6.pkl'
    saved_dataset_path = './dataset/dataset_for_tl/t6/t6_train_ssledge1.pkl'
    dataset_edge_sampler(dataset_path, saved_dataset_path)