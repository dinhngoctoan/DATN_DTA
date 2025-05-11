import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GIN_GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GIN_GCNNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        #3D protein
        self.conv_prot1 = GCNConv(num_features_xt, num_features_xt)
        self.conv_prot2 = GCNConv(num_features_xt, 64)
        self.lin1 = nn.Linear(64,output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data_drug, data_protein):
        x_drug, edge_index_drug, batch_drug = data_drug.x, data_drug.edge_index, data_drug.batch

        x_drug = F.relu(self.conv1(x_drug, edge_index_drug))
        x_drug = self.bn1(x_drug)
        x_drug = F.relu(self.conv2(x_drug, edge_index_drug))
        x_drug = self.bn2(x_drug)
        x_drug = F.relu(self.conv3(x_drug, edge_index_drug))
        x_drug = self.bn3(x_drug)
        x_drug = F.relu(self.conv4(x_drug, edge_index_drug))
        x_drug = self.bn4(x_drug)
        x_drug = F.relu(self.conv5(x_drug, edge_index_drug))
        x_drug = self.bn5(x_drug)
        x_drug = global_add_pool(x_drug, batch_drug)
        x_drug = F.relu(self.fc1_xd(x_drug))
        x_drug = F.dropout(x_drug, p=0.2, training=self.training)

        x_prots, edge_index_prots, prots_batch= data_protein.x, data_protein.edge_index, data_protein.batch
        x_prots = F.dropout(x_prots, p=0.2, training=self.training)
        x_prots = self.relu(self.conv_prot1(x_prots,edge_index_prots))
        x_prots = F.dropout(x_prots, p=0.2, training=self.training)
        x_prots = self.relu(self.conv_prot2(x_prots,edge_index_prots))
        x_prots = gmp(x_prots, prots_batch)
        x_prots = self.lin1(x_prots)
        x_prots = self.relu(x_prots)

        # concat
        xc = torch.cat((x_drug, x_prots), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
