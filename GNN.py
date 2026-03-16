import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GCNConv, SAGEConv, SGConv

class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GCNClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class SAGEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(SAGEClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class SGCClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(SGCClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv = SGConv(input_dim, output_dim, K=2, add_self_loops = False) # already has self-loops

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv(x, edge_index)
        return x

class APPNPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(APPNPClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.propagate = APPNP(K=2, alpha=0.1, dropout=dropout, add_self_loops = False) # alread has self-loops

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = self.propagate(x, edge_index)
        return x
