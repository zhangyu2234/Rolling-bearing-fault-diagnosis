import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F

class GCN_Pyg(nn.Module):
    def __init__(self, in_channel, hidden, out_channel, dropout):
        super(GCN_Pyg, self).__init__()

        self.in_channel = in_channel
        self.hidden = hidden
        self.out_channel = out_channel
        self.dropout = dropout

        self.layer1 = GCNConv(in_channel, hidden)
        self.layer2 = GCNConv(hidden, out_channel)

    
    def forward(self, x, adj, edge_weights):

        x = F.relu(self.layer1(x, adj, edge_weights))

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layer2(x, adj, edge_weights)
        return F.log_softmax(x, dim=-1)
    

    


    

