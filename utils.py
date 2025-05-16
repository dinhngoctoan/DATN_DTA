import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
#from ProtDrugData import DrugProteinData
from torch_geometric.data import Batch

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',type=None, 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.type = type
 
        if os.path.isfile(self.processed_paths[0]):
            print(self.processed_paths[0], 'exists')
            self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
            print('Data processed and loaded successfully')

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        if self.type == 'drug':
            return [self.dataset + '_drug.pt']
        elif self.type == 'protein':
            return [self.dataset + '_protein.pt']
        else:
            return [self.dataset + '.pt']  # Fallback mặc định

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
    # Inputs:
    # XD - list of SMILES, XT: list of Protein with 3D,
    # Y: list of labels (i.e. affinity)

    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        if self.type == 'drug':
            for i in range(data_len):
                print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
                smiles = xd[i]
                labels = y[i]
            
            # Convert SMILES to molecular representation using rdkit
                c_size, features, edge_index = smile_graph[smiles]
            
            # Create drug data using PyTorch Geometric Data class
                drug_data = DATA.Data(
                    x=torch.tensor(features, dtype=torch.float),
                    edge_index=torch.tensor(np.array(edge_index).astype(np.int64), dtype=torch.long).transpose(1, 0),
                    c_size=torch.tensor([c_size], dtype=torch.long),
                    y = torch.tensor([labels], dtype=torch.float)
                )
                data_list.append(drug_data)
        elif self.type == 'protein':
            for i in range(data_len):
                print('Converting Protein to graph: {}/{}'.format(i+1, data_len))
                target = xt[i]
            # Unpack the target tuple
                target_node_features, target_edge_index, target_edge_features = target
            
            # Create protein data using PyTorch Geometric Data class
                protein_data = DATA.Data(
                x=torch.tensor(target_node_features, dtype=torch.float),
                edge_index=torch.tensor(np.array(target_edge_index).astype(np.int64), dtype=torch.long).transpose(1, 0),
                edge_attr=torch.tensor(target_edge_features, dtype=torch.float)
                )
                data_list.append(protein_data)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        print('Graph construction done. Preparing data for saving.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
 
        print('Data saved successfully.')


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
