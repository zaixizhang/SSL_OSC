import dgl
import torch
import pickle
import numpy as np
import time
import pickle
import networkx as nx

def edge_pair_to_index(pair, num_atoms):
    i, j = pair[0], pair[1]
    return i * num_atoms + j


def random_sample_atoms(g, rate_list):
    num_atoms = len(g.nodes())
    n_layers = len(rate_list)

    sample_num_list = []
    sample_num_list.append(int(num_atoms*rate_list[0]))
    for i in range(1, n_layers):
        sample_num_list.append(int(sample_num_list[i-1]*rate_list[i]))
    
    atom_index_list = []
    atom_index_list.append(range(num_atoms))
    atom_index_list.append(np.random.choice(num_atoms, sample_num_list[0], replace=False))
    for i in range(1, n_layers):
        atom_index_list.append(np.random.choice(atom_index_list[i-1], sample_num_list[i], replace=False))
    
    return atom_index_list

hartree_to_ev = 27.2116

map_U0 = {
    1 : -0.500273,
    6 : -37.846772,
    7 : -54.583861,
    8 : -75.064579,
    9 : -99.718730
}

map_dict = {
    'U0' : map_U0
}

def batched_ref(node_type_list, num_of_atom_list, hartree_flag=False, property_id='U0'):
    map = map_dict[property_id]
    batch_size = len(num_of_atom_list)

    molecule_atom_type_list = []
    batched_ref = []
    sum = 0
    for i in num_of_atom_list:
        temp = node_type_list[sum:sum+i]
        sum += i
        molecule_atom_type_list.append(temp)
    
    for i in range(batch_size):
        ref_value = 0.0
        molecule_i_atom_type = molecule_atom_type_list[i]
        for j in molecule_i_atom_type:
            ref_value += map[j]
        batched_ref.append(ref_value)
    
    if hartree_flag == False:
        batched_ref = [i * hartree_to_ev for i in batched_ref]
    batched_ref = [[i] for i in batched_ref]

    return batched_ref


# X is the batch data delete edges (dgl.DGLGraph)
# delta is threshold value
def delete_delta_edges(X, delta):
    x_pool = []

    x_list = dgl.unbatch(X)
    for x_i in x_list:
        distance_list = x_i.edata['distance'].view(-1,1)
        for index in range(len(distance_list)):
            if distance_list[index] > delta:
                x_i.remove_edges(index)
        x_pool.append(x_i)

    X_ = dgl.batch(x_pool)
    return X_

class Logger():
    def __init__(self, paras_dict, logger_flag=True, filename=None):
        if filename == None:
            self.filename = log_file_path + time.strftime("%Y-%m-%d-%H_%M", time.localtime())
        else:
            self.filename = log_file_path + filename
        self.paras_dict = paras_dict
        self.logger_flag = logger_flag

        if self.logger_flag:
            self._record_parameters()

    def _record_parameters(self):
        f = open(self.filename, 'a')
        keys = list(self.paras_dict.keys())
        values = list(self.paras_dict.values())

        for iteration in range(len(keys)):
            key = keys[iteration]
            value = values[iteration]

            infor = key + ' : ' + str(value)
            f.write(infor)
            f.write('\n')
        f.close()

    def append_record(self, infor):
        if self.logger_flag:
            f = open(self.filename, 'a')
            f.write(infor)
            f.write('\n')
            f.close()

class AtomRandomSampler():
    def __init__(self, data, num_atom_list):
        self.data = data
        self.num_atom_list = num_atom_list

    def get_atom_list(self):
        data_list = []
        for data_i in self.data:
            data_i_atom_list = []
            num_atoms = len(data_i.nodes())
            list0 = np.random.choice(num_atoms, self.num_atom_list[0], replace=False)
            data_i_atom_list.append(list0)

            for i in range(1, len(self.num_atom_list)):
                list_i = np.random.choice(data_i_atom_list[i-1], self.num_atom_list[i], replace=False)
                data_i_atom_list.append(list_i)
            data_list.append(data_i_atom_list)
        return data_list


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def rbf_dot(X, Y, deg):
    # X ~ [m, feature_dim]
    # Y ~ [n, feature_dim]
    # deg is the sigma of guassian kernel

    nx = len(X)
    ny = len(Y)

    G = torch.sum(torch.mul(X, X), 1).view(-1,1)
    H = torch.sum(torch.mul(Y, Y), 1).view(1,-1)
    
    Q = G.repeat(1, ny)
    R = H.repeat(nx, 1)

    H = Q + R - 2 * torch.mm(X, Y.t())
    H = H/2.0
    H = H/(deg*deg)
    return torch.exp(-H)

def mmd_XY(X, Y, sigma):
    nx = len(X)
    ny = len(Y)

    K = rbf_dot(X, X, sigma)
    L = rbf_dot(Y, Y, sigma)
    KL = rbf_dot(X, Y, sigma)
    c_K = 1/(nx*nx)
    c_L = 1/(ny*ny)
    c_KL = 2/(nx*ny)
    return torch.sqrt(torch.sum(c_K*K) + torch.sum(c_L*L) - torch.sum(c_KL*KL))

def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


if __name__ == '__main__':
    mol = pickle.load(open('./test_mol.pkl', 'rb'))
    random_sample_atoms(mol, [0.6, 0.6])