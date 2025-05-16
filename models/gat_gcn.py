import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.nn import global_mean_pool as gMeanp
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import NNConv
# GAT  model
class GAT_GCNNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=41,
                      output_dim=128, dropout=0.2,
                     ):
        super(GAT_GCNNet, self).__init__()

        # drug
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.conv2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        #3D protein
        self.conv_prot1 = GCNConv(num_features_xt, num_features_xt)
        self.conv_prot2 = GCNConv(num_features_xt, 64)
        self.lin1 = nn.Linear(64,output_dim)

        #concat
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data_drug,data_protein):
        # graph input feed-forward
        x_drug, edge_index_drug, batch_drug = data_drug.x, data_drug.edge_index, data_drug.batch

        x_drug = F.dropout(x_drug, p=0.2, training=self.training)
        x_drug = F.elu(self.conv1(x_drug, edge_index_drug))
        x_drug = F.dropout(x_drug, p=0.2, training=self.training)
        x_drug = self.conv2(x_drug, edge_index_drug)
        x_drug = self.relu(x_drug)
        x_drug = gmp(x_drug, batch_drug)          
        x_drug = self.fc_g1(x_drug)
        x_drug = self.relu(x_drug)

        # protein input feed-forward:
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
        #dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
