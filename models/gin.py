import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv,GCNConv, GINConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
# GAT  model
class GIN_Model(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=41,num_features_xt_seq = 25,
                      output_dim=128, dropout=0.2,embed_dim=128,n_filters=32):
        super(GIN_Model, self).__init__()
        dim = 32
        # drug
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

        self.fc1_xd = Linear(dim*2, output_dim)

        #ecfp
        self.fc_ecfp = nn.Linear(2048,128)
        self.bn_ecfp = nn.BatchNorm1d(128)
        #3D protein
        self.conv_prot1 = GCNConv(num_features_xt, num_features_xt)
        self.conv_prot2 = GCNConv(num_features_xt, 64)
        self.lin1 = nn.Linear(64,128)

        #1D protein
        self.embedding_xt = nn.Embedding(num_features_xt_seq + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(2,ceil_mode=True)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(2,ceil_mode=True)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(2,ceil_mode=True)
        self.fc_xt = nn.Linear(10*128, 128)
        
        #cross-attention
        self.drug_protein_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=dropout)
        self.protein_drug_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=dropout)
        
        #concat
        self.fc_xd = nn.Linear(256,128)
        self.fc_xp = nn.Linear(256,128)

        self.fc1 = nn.Linear(128*2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data_drug,data_protein):
        # graph input feed-forward
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
        x_drug = torch.cat([gmp(x_drug, batch_drug), gap(x_drug, batch_drug)], dim=1)
        x_drug = F.relu(self.fc1_xd(x_drug))
        x_drug = F.dropout(x_drug, p=0.2, training=self.training)
        #ecfp
        x_drug_ecfp = data_drug.ecfp
        x_drug_ecfp = self.fc_ecfp(x_drug_ecfp)
        x_drug_ecfp = self.bn_ecfp(x_drug_ecfp)     
        x_drug_ecfp = self.relu(x_drug_ecfp)
        x_drug_ecfp = self.dropout(x_drug_ecfp)
        
        # protein input feed-forward:
        x_prots, edge_index_prots, prots_batch= data_protein.x, data_protein.edge_index, data_protein.batch
        #3D process 
        x_prots = F.dropout(x_prots, p=0.2, training=self.training)
        x_prots = self.relu(self.conv_prot1(x_prots,edge_index_prots))
        x_prots = F.dropout(x_prots, p=0.2, training=self.training)
        x_prots = self.relu(self.conv_prot2(x_prots,edge_index_prots))
        x_prots = gmp(x_prots, prots_batch)
        x_prots = self.lin1(x_prots)
        x_prots = self.relu(x_prots)
        #1D process
        prot_seq = data_drug.protein_seq
        embedded_x_seq = self.embedding_xt(prot_seq)
        x_prots_seq = self.conv_xt_1(embedded_x_seq)
        x_prots_seq = F.relu(x_prots_seq)
        x_prots_seq = self.pool_xt_1(x_prots_seq)
        x_prots_seq = self.conv_xt_2(x_prots_seq)
        x_prots_seq = F.relu(x_prots_seq)
        x_prots_seq = self.pool_xt_2(x_prots_seq)
        x_prots_seq = self.conv_xt_3(x_prots_seq)
        x_prots_seq = F.relu(x_prots_seq)
        x_prots_seq = self.pool_xt_3(x_prots_seq)

        # flatten
        x_prots_seq = x_prots_seq.view(-1, 10 * 128)
        x_prots_seq = self.fc_xt(x_prots_seq)

        #cross-attention
        x_drug = x_drug.unsqueeze(0)  
        x_prots = x_prots.unsqueeze(0)  
        
        # Drug to protein attention
        x_drug_attended, _ = self.drug_protein_attention(x_drug, x_prots, x_prots)
        x_drug_attended = x_drug_attended.squeeze(0)  
        
        # Protein to drug attention
        x_prots_attended, _ = self.protein_drug_attention(x_prots, x_drug, x_drug)
        x_prots_attended = x_prots_attended.squeeze(0)  

        # concat
        xd = torch.cat((x_drug_attended,x_drug_ecfp),1)
        xp = torch.cat((x_prots_attended,x_prots_seq),1)
        xd = self.fc_xd(xd)
        xd = self.relu(xd)
        xp = self.fc_xp(xp)
        xp = self.relu(xp)
        
        xc = torch.cat((xd,xp), 1)
        #dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
