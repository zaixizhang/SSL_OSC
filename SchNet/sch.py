import dgl
import torch as th
import torch.nn as nn
from layers import AtomEmbedding, Interaction, ShiftSoftplus, RBFLayer
import torch
from torch.autograd  import  Function
import numpy as np

class SchNetModel(nn.Module):
    """
    SchNet Model from:
        Schütt, Kristof, et al.
        SchNet: A continuous-filter convolutional neural network
        for modeling quantum interactions. (NIPS'2017)
    """

    def __init__(self,
                 dim=64,
                 cutoff=5.0,
                 output_dim=1,
                 width=0.1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None,
                 mmd_flag=False):
        """
        Args:
            dim: dimension of features
            output_dim: dimension of prediction
            cutoff: radius cutoff
            width: width in the RBF function
            n_conv: number of interaction layers
            atom_ref: used as the initial value of atom embeddings,
                      or set to None with random initialization
            norm: normalization
        """
        super().__init__()
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm
        self.activation = ShiftSoftplus()
        self.mmd_flag = mmd_flag

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)
        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)
        self.rbf_layer = RBFLayer(0, cutoff, width, 1)
        self.conv_layers = nn.ModuleList(
            [Interaction(self.rbf_layer._fan_out, dim) for i in range(n_conv)])

        self.atom_dense_layer1 = nn.Linear(dim, 64)
        self.atom_dense_layer2 = nn.Linear(64, output_dim)

    def set_mean_std(self, mean, std, device="cpu"):
        self.mean_per_atom = th.tensor(mean, device=device)
        self.std_per_atom = th.tensor(std, device=device)

    def forward(self, g):
        """g is the DGL.graph"""

        self.embedding_layer(g)
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)
        for idx in range(self.n_conv):
            self.conv_layers[idx](g)

        feature = dgl.mean_nodes(g, "node")
        atom = self.atom_dense_layer1(g.ndata["node"])
        atom = self.activation(atom)
        res = self.atom_dense_layer2(atom)
        g.ndata["res"] = res

        if self.atom_ref is not None:
            g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]

        # if self.norm:
        #     g.ndata["res"] = g.ndata[
        #         "res"] * self.std_per_atom + self.mean_per_atom
        res = dgl.mean_nodes(g, "res")
        if not self.mmd_flag:
            return res
        else:
            return res, feature

class SchNetModelGRL(nn.Module):
    def __init__(self,
                cutoff=3, 
                width=0.1, 
                n_conv=2):

        super().__init__()
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv

        self.schnet = SchNetModel(cutoff=self.cutoff, width=self.width, n_conv=self.n_conv, mmd_flag=True)
        self.control_net = ControlNet(self._dim)

    def forward(self, g):
        res, feature = self.schnet(g)
        domain_prob = self.control_net(feature)
        return res, domain_prob



class GRL(Function):
    def forward(self,input):
        return input
    def backward(self,grad_output):
        grad_input = grad_output.neg()
        return grad_input


class ControlNet(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

        self.feature_dense_layer1 = nn.Linear(self.dim, 128)
        self.feature_dense_layer2 = nn.Linear(128, 64)
        self.feature_dense_layer3 = nn.Linear(64, 32)
        self.feature_dense_layer4 = nn.Linear(32, 2)

    def forward(self, feature):
        h1 = self.feature_dense_layer1(feature)
        h2 = self.feature_dense_layer2(h1)
        h3 = self.feature_dense_layer3(h2)
        h4 = self.feature_dense_layer4(h3)

        return h4

class SSLSchNetModel(nn.Module):
    """
    SchNet Model from:
        Schütt, Kristof, et al.
        SchNet: A continuous-filter convolutional neural network
        for modeling quantum interactions. (NIPS'2017)
    """

    def __init__(self,
                 dim=64,
                 cutoff=5.0,
                 output_dim=1,
                 width=0.1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None,
                 mmd_flag=False):
        """
        Args:
            dim: dimension of features
            output_dim: dimension of prediction
            cutoff: radius cutoff
            width: width in the RBF function
            n_conv: number of interaction layers
            atom_ref: used as the initial value of atom embeddings,
                      or set to None with random initialization
            norm: normalization
        """
        super().__init__()
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm
        self.activation = ShiftSoftplus()
        self.mmd_flag = mmd_flag

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)
        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)
        self.rbf_layer = RBFLayer(0, cutoff, width, 1)
        self.edge_mask = nn.Parameter(torch.zeros(int(np.ceil(cutoff / width))), requires_grad=True)
        self.conv_layers = nn.ModuleList(
            [Interaction(self.rbf_layer._fan_out, dim) for i in range(n_conv)])

        self.atom_dense_layer1 = nn.Linear(dim, 64)
        self.atom_dense_layer2 = nn.Linear(64, output_dim)

        self.node_type_layer1 = nn.Linear(64, 32)
        self.node_type_layer2 = nn.Linear(32, 3)

        self.edge_type_layer1 = nn.Linear(128, 64)
        self.edge_type_layer2 = nn.Linear(64, 5)

        #self.control_layer = ControlNet(self._dim)
        #self.grl = GRL()

    def set_mean_std(self, mean, std, device="cpu"):
        self.mean_per_atom = th.tensor(mean, device=device)
        self.std_per_atom = th.tensor(std, device=device)

    def forward(self, g, node_index, source_index, target_index, select_edge_index):
        """g is the DGL.graph"""

        self.embedding_layer(g)
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)
        g.edata["rbf"][select_edge_index.view(-1).long()] = self.edge_mask
        for idx in range(self.n_conv):
            self.conv_layers[idx](g)

        feature = g.ndata["node"]
        feature_ = dgl.mean_nodes(g, "node")

        # for node_loss
        node_select_feature = feature.index_select(0, node_index.view(-1).long())
        node_select_feature = self.node_type_layer1(node_select_feature)
        node_type = self.node_type_layer2(node_select_feature)

        # for edge_loss
        source_select_feature = feature.index_select(0, source_index.view(-1).long())
        target_select_feature = feature.index_select(0, target_index.view(-1).long())
        edge_feature = torch.cat([source_select_feature, target_select_feature], 1)
        edge_type = self.edge_type_layer1(edge_feature)
        edge_type = self.edge_type_layer2(edge_type)

        # for domain loss
        #domain_prob = self.control_layer(feature_)

        return node_type, edge_type