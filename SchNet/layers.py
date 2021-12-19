import torch as th
import numpy as np
import torch.nn as nn
import dgl.function as fn
from torch.nn import Softplus

class AtomEmbedding(nn.Module):
    """
    Convert the atom(node) list to atom embeddings.
    The atom with the same element share the same initial embeddding.
    """

    def __init__(self, dim=128, type_num=100, pre_train=None):
        """
        Randomly init the element embeddings.
        Args:
            dim: the dim of embeddings
            type_num: the largest atomic number of atoms in the dataset
            pre_train: the pre_trained embeddings
        """
        super().__init__()
        self._dim = dim
        self._type_num = type_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train,
                                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding(type_num, dim, padding_idx=0)

    def forward(self, g, p_name="node"):
        """Input type is dgl graph"""
        atom_list = g.ndata["node_type"]
        g.ndata[p_name] = self.embedding(atom_list)
        return g.ndata[p_name]

class ShiftSoftplus(Softplus):
    """
    Shiftsoft plus activation function:
        1/beta * (log(1 + exp**(beta * x)) - log(shift))
    """

    def __init__(self, beta=1, shift=2, threshold=20):
        super().__init__(beta, threshold)
        self.shift = shift
        self.softplus = Softplus(beta, threshold)

    def forward(self, input):
        return self.softplus(input) - np.log(float(self.shift))

class RBFLayer(nn.Module):
    """
    Radial basis functions Layer.
    e(d) = exp(- gamma * ||d - mu_k||^2)
    default settings:
        gamma = 10
        0 <= mu_k <= 30 for k=1~300
    """

    def __init__(self, low=0, high=30, gap=0.1, dim=1, name='rbf'):
        super().__init__()
        self._low = low
        self._high = high
        self._gap = gap
        self._dim = dim
        self.name = name

        self._n_centers = int(np.ceil((high - low) / gap))
        centers = np.linspace(low, high, self._n_centers)
        self.centers = th.tensor(centers, dtype=th.float, requires_grad=False)
        self.centers = nn.Parameter(self.centers, requires_grad=False)
        self._fan_out = self._dim * self._n_centers

        self._gap = centers[1] - centers[0]

    def dis2rbf(self, edges):
        dist = edges.data["distance"]
        radial = dist - self.centers
        coef = -1 / self._gap
        rbf = th.exp(coef * (radial**2))
        return {self.name: rbf}

    def forward(self, g):
        """Convert distance scalar to rbf vector"""
        g.apply_edges(self.dis2rbf)
        return g.edata[self.name]


class CFConv(nn.Module):
    """
    The continuous-filter convolution layer in SchNet.
    One CFConv contains one rbf layer and three linear layer
        (two of them have activation funct).
    """

    def __init__(self, rbf_dim, dim=64, act="sp"):
        """
        Args:
            rbf_dim: the dimsion of the RBF layer
            dim: the dimension of linear layers
            act: activation function (default shifted softplus)
        """
        super().__init__()
        self._rbf_dim = rbf_dim
        self._dim = dim

        self.linear_layer1 = nn.Linear(self._rbf_dim, self._dim)
        self.linear_layer2 = nn.Linear(self._dim, self._dim)

        if act == "sp":
            self.activation = nn.Softplus(beta=0.5, threshold=14)
        else:
            self.activation = act

    def update_edge(self, edges):
        rbf = edges.data["rbf"]
        h = self.linear_layer1(rbf)
        h = self.activation(h)
        h = self.linear_layer2(h)
        return {"h": h}

    def forward(self, g):
        g.apply_edges(self.update_edge)
        #print(g.edata['h'].size())
        #rint(g.ndata['new_node'].size())
        g.update_all(message_func=fn.u_mul_e('new_node', 'h', 'neighbor_info'),
                     reduce_func=fn.sum('neighbor_info', 'new_node'))
        return g.ndata["new_node"]

class CFConvCluster(nn.Module):
    def __init__(self, rbf_dim, dim=64, act="sp", diff_rate=0.2):
        
        super().__init__()

        self._rbf_dim = rbf_dim
        self._dim = dim
        self.diff_rate = diff_rate

        self.linear_layer1 = nn.Linear(self._rbf_dim, self._dim)
        self.linear_layer2 = nn.Linear(self._dim, self._dim)

        if act == "sp":
            self.activation = nn.Softplus(beta=0.5, threshold=14)
        else:
            self.activation = act


    def filter_edge_cluster(self, edges):
        return (edges.src['cluster_id'] == 0) * (edges.dst['cluster_id'] == 0)

    
    def filter_diff_edge(self, edges):
        return (edges.src['cluster_id'] != edges.dst['cluster_id'])


    def filter_func(self, cluster_id):
        def filter_edge_cluster(edges):
            return (edges.src['cluster_id'] == cluster_id) * (edges.dst['cluster_id'] == cluster_id)
        return filter_edge_cluster

    def update_edge(self, edges):
        rbf = edges.data["rbf"]
        h = self.linear_layer1(rbf)
        h = self.activation(h)
        h = self.linear_layer2(h)
        return {"h": h}
    
    def send_func(self, edges):
        return {'neighbor_info' : edges.src['new_node'] * edges.data['h']}
    
    def recv_func(self, nodes):
        return {'new_node' : nodes.mailbox['neighbor_info'].sum(1)}

    def forward(self, g, cluster_id):
        g.apply_edges(self.update_edge)
        #print(g.edata['h'].size())
        #print(g.ndata['new_node'].size())
        # g.update_all(message_func=fn.u_mul_e('new_node', 'h', 'neighbor_info'),
        #              reduce_func=fn.sum('neighbor_info', 'new_node'))
        # return g.ndata["new_node"]


        g.register_message_func(self.send_func)
        g.register_reduce_func(self.recv_func)

        diff_edge_index = g.filter_edges(self.filter_diff_edge).tolist()
        cluster_index = list(set(range(len(g.edges()[0]))) - set(diff_edge_index))

        g.send(cluster_index)

        index = np.random.choice(diff_edge_index, int(len(diff_edge_index) * self.diff_rate), replace=False)
        g.send(index)

        g.recv(g.nodes())
        return g.ndata["new_node"]


class InteractionCluster(nn.Module):
    def __init__(self, rbf_dim, dim, diff_rate=0.2):
        super().__init__()
        self._node_dim = dim
        self.activation = nn.Softplus(beta=0.5, threshold=14)
        self.node_layer1 = nn.Linear(dim, dim, bias=False)
        self.cfconv = CFConvCluster(rbf_dim, dim, act=self.activation, diff_rate=diff_rate)
        self.node_layer2 = nn.Linear(dim, dim)
        self.node_layer3 = nn.Linear(dim, dim)

    def forward(self, g, num_cluster):
        g.ndata["new_node"] = self.node_layer1(g.ndata["node"])
        cf_node = self.cfconv(g, num_cluster)
        cf_node_1 = self.node_layer2(cf_node)
        cf_node_1a = self.activation(cf_node_1)
        new_node = self.node_layer3(cf_node_1a)
        g.ndata["node"] = g.ndata["node"] + new_node
        return g.ndata["node"]

class Interaction(nn.Module):
    """
    The interaction layer in the SchNet model.
    """

    def __init__(self, rbf_dim, dim):
        super().__init__()
        self._node_dim = dim
        self.activation = nn.Softplus(beta=0.5, threshold=14)
        self.node_layer1 = nn.Linear(dim, dim, bias=False)
        self.cfconv = CFConv(rbf_dim, dim, act=self.activation)
        self.node_layer2 = nn.Linear(dim, dim)
        self.node_layer3 = nn.Linear(dim, dim)

    def forward(self, g):

        g.ndata["new_node"] = self.node_layer1(g.ndata["node"])
        cf_node = self.cfconv(g)
        cf_node_1 = self.node_layer2(cf_node)
        cf_node_1a = self.activation(cf_node_1)
        new_node = self.node_layer3(cf_node_1a)
        g.ndata["node"] = g.ndata["node"] + new_node
        return g.ndata["node"]