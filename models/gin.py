import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GATConv, GCNConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp

# Hybrid ligand GAT/GCN and protein GIN model.
class GIN_Model(torch.nn.Module):
    def __init__(self, data_store, num_features_xd=78, n_output=1, num_features_xt=41,
                      output_dim=128, dropout=0.2, protein_embedding_dim=640):
        super().__init__()
        # The store is a plain Python object; state_dict contains only model weights.
        self.data_store = data_store
        if output_dim != 128:
            raise ValueError('Fusion requires output_dim=128')

        dim = 32

        # Ligand graph: GAT 8x64 -> GCN 256, followed by max+mean pooling.
        self.conv1 = GATConv(num_features_xd, 64, heads=8, concat=True)
        self.conv2 = GCNConv(512, 256)


        self.fc_g1 = torch.nn.Linear(512, 256)
        self.fc_g2 = torch.nn.Linear(256, output_dim)

        # Replace ECFP with Chemformer ligand embeddings. Pool only in forward.
        self.fc_drug_embedding = nn.Linear(512, 128)
        self.bn_drug_embedding = nn.BatchNorm1d(128)

        # Protein graph: five GIN layers with 41 input features and separate weights.
        nn_p1 = Sequential(Linear(num_features_xt, dim), ReLU(), Linear(dim, dim))
        self.conv_prot1 = GINConv(nn_p1)
        self.bn_prot1 = nn.BatchNorm1d(dim)

        nn_p2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv_prot2 = GINConv(nn_p2)
        self.bn_prot2 = nn.BatchNorm1d(dim)

        nn_p3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv_prot3 = GINConv(nn_p3)
        self.bn_prot3 = nn.BatchNorm1d(dim)

        nn_p4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv_prot4 = GINConv(nn_p4)
        self.bn_prot4 = nn.BatchNorm1d(dim)

        nn_p5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv_prot5 = GINConv(nn_p5)
        self.bn_prot5 = nn.BatchNorm1d(dim)

        self.fc_prot_g1 = nn.Linear(dim, 256)
        self.fc_prot_g2 = nn.Linear(256, output_dim)

        # Protein ESM-2: concatenate max/mean pooling, then project 1280 -> 128.
        self.fc_protein_embedding = nn.Linear(2 * protein_embedding_dim, 128)
        self.bn_protein_embedding = nn.BatchNorm1d(128)

        #concat
        self.fc_xd = nn.Linear(256,128)
        self.fc_xp = nn.Linear(256,128)

        # Small prediction head: 256 -> 512 -> 128 -> affinity.
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_indices):
        # Look up all four sources on the GPU at the start of each forward pass.
        data_drug, data_protein = self.data_store.lookup(pair_indices)
        # graph input feed-forward
        x_drug, edge_index_drug, batch_drug = data_drug.x, data_drug.edge_index, data_drug.batch

        # Ligand graph: GAT -> ReLU -> GCN -> ReLU.
        x_drug = self.conv1(x_drug, edge_index_drug)
        x_drug = self.relu(x_drug)
        x_drug = self.conv2(x_drug, edge_index_drug)
        x_drug = self.relu(x_drug)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x_drug = torch.cat([gmp(x_drug, batch_drug), gap(x_drug, batch_drug)], dim=1)
        x_drug = self.relu(self.fc_g1(x_drug))
        x_drug = self.dropout(x_drug)
        x_drug = self.fc_g2(x_drug)

        # Tokens are concatenated; drug_embedding_batch identifies their samples.
        # The bank preserves raw tokens; max pooling happens only here.
        x_drug_embedding = gmp(data_drug.drug_embedding, data_drug.drug_embedding_batch)
        x_drug_embedding = self.fc_drug_embedding(x_drug_embedding)
        # BatchNorm needs >=2 training samples; use running statistics for a singleton batch.
        if self.training and x_drug_embedding.size(0) == 1:
            x_drug_embedding = F.batch_norm(
                x_drug_embedding, self.bn_drug_embedding.running_mean,
                self.bn_drug_embedding.running_var, self.bn_drug_embedding.weight,
                self.bn_drug_embedding.bias, training=False, eps=self.bn_drug_embedding.eps)
        else:
            x_drug_embedding = self.bn_drug_embedding(x_drug_embedding)
        x_drug_embedding = self.relu(x_drug_embedding)
        x_drug_embedding = self.dropout(x_drug_embedding)

        # protein input feed-forward:
        x_prots, edge_index_prots, prots_batch= data_protein.x, data_protein.edge_index, data_protein.batch
        # Protein graph: five GIN layers, each followed by ReLU and BatchNorm.
        x_prots = self.bn_prot1(self.relu(self.conv_prot1(x_prots, edge_index_prots)))
        x_prots = self.bn_prot2(self.relu(self.conv_prot2(x_prots, edge_index_prots)))
        x_prots = self.bn_prot3(self.relu(self.conv_prot3(x_prots, edge_index_prots)))
        x_prots = self.bn_prot4(self.relu(self.conv_prot4(x_prots, edge_index_prots)))
        x_prots = self.bn_prot5(self.relu(self.conv_prot5(x_prots, edge_index_prots)))
        x_prots = global_add_pool(x_prots, prots_batch)
        x_prots = self.relu(self.fc_prot_g1(x_prots))
        x_prots = self.dropout(x_prots)
        x_prots = self.fc_prot_g2(x_prots)

        # Pool tokens per sample using protein_embedding_batch, not the graph node batch.
        # Concatenation order: [max (640), mean (640)] -> [B,1280].
        x_protein_embedding = torch.cat([
            gmp(data_protein.protein_embedding, data_protein.protein_embedding_batch),
            gap(data_protein.protein_embedding, data_protein.protein_embedding_batch),
        ], dim=1)
        x_protein_embedding = self.fc_protein_embedding(x_protein_embedding)
        if self.training and x_protein_embedding.size(0) == 1:
            x_protein_embedding = F.batch_norm(
                x_protein_embedding, self.bn_protein_embedding.running_mean,
                self.bn_protein_embedding.running_var, self.bn_protein_embedding.weight,
                self.bn_protein_embedding.bias, training=False, eps=self.bn_protein_embedding.eps)
        else:
            x_protein_embedding = self.bn_protein_embedding(x_protein_embedding)
        x_protein_embedding = self.relu(x_protein_embedding)
        x_protein_embedding = self.dropout(x_protein_embedding)

        # Ablation: concatenate graph features directly, without cross-attention or residuals.
        xd = torch.cat((x_drug, x_drug_embedding), dim=1)       # [B,256]
        xp = torch.cat((x_prots, x_protein_embedding), dim=1)  # [B,256]
        xd = self.relu(self.fc_xd(xd))
        xp = self.relu(self.fc_xp(xp))

        xc = torch.cat((xd,xp), 1)
        # Apply ReLU/Dropout after each hidden layer; the output layer is linear.
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
