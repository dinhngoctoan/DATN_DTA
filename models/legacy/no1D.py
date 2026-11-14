import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv,GCNConv, GINConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import NNConv
# GAT  model
class no1D_Model(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=41,num_features_xt_seq = 25,
                      output_dim=128, dropout=0.2,embed_dim=128,n_filters=32):
        super(no1D_Model, self).__init__()
        dim = 32
        # drug
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)

        #ecfp
        self.fc_ecfp = nn.Linear(2048,128)
        self.bn_ecfp = nn.BatchNorm1d(128)
        #3D protein
        self.conv_prot1 = GCNConv(num_features_xt, num_features_xt)
        self.conv_prot2 = GCNConv(num_features_xt, 64)
        self.lin1 = nn.Linear(64,128)

        
        #cross-attention
        #3D
        self.drug_protein_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=dropout)
        self.protein_drug_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=dropout)
        #1D
        self.drugecfp_protein_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=dropout)
        self.protein_drugecfp_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=dropout)        
        #concat
        self.fc_xd = nn.Linear(256,128)
        self.fc_xp = nn.Linear(256,128)

        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data_drug,data_protein):
        # graph input feed-forward
        x_drug, edge_index_drug, batch_drug = data_drug.x, data_drug.edge_index, data_drug.batch

        x_drug = self.conv1(x_drug, edge_index_drug)
        x_drug = self.relu(x_drug)
        x_drug = self.conv2(x_drug, edge_index_drug)
        x_drug = self.relu(x_drug)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x_drug = torch.cat([gmp(x_drug, batch_drug), gap(x_drug, batch_drug)], dim=1)
        x_drug = self.relu(self.fc_g1(x_drug))
        x_drug = self.dropout(x_drug)
        x_drug = self.fc_g2(x_drug)

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
  


        # concat
        #xd = torch.cat((x_drug,x_drug_ecfp),1)
        #xp = torch.cat((x_prots,x_prots_seq),1)
        #cross-attention
        x_drug = x_drug.unsqueeze(0)  # [1, batch_size, embed_dim]
        x_prots = x_prots.unsqueeze(0)  # [1, batch_size, embed_dim]
        x_drug_ecfp = x_drug_ecfp.unsqueeze(0)
        # Drug to protein attention
        xd_attended_1, _ = self.drug_protein_attention(x_drug, x_prots, x_prots)
        xd_attended_1 = xd_attended_1.squeeze(0) 

        xd_attended_2, _ = self.drugecfp_protein_attention(x_drug_ecfp, x_prots, x_prots)
        xd_attended_2 = xd_attended_2.squeeze(0)  
        # Protein to drug attention
        xp_attended_3d, _ = self.protein_drug_attention(x_prots, x_drug, x_drug)
        xp_attended_3d = xp_attended_3d.squeeze(0)  
        
        xp_attended_ecfp, _ = self.protein_drugecfp_attention(x_prots, x_drug_ecfp, x_drug_ecfp)
        xp_attended_ecfp = xp_attended_ecfp.squeeze(0) 

        xp_attended = xp_attended_3d+xp_attended_ecfp
        xc = torch.cat((xd_attended_1,xd_attended_2,xp_attended), 1)
        #dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out