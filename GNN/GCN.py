import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN.GCN_layer import GCN_layer


class GCN(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, dropout=0.2):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.layer1 = GCN_layer(in_channel, hidden_channel)
        self.layer2 = GCN_layer(hidden_channel, out_channel)

    
    def forward(self, x, adj):
        x = F.relu(self.layer1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.layer2(x, adj))
        return F.log_softmax(x, dim=-1)
    

if __name__ == '__main__':

    x = torch.randn(5, 3)
    adj = torch.randn(5, 5)
    model = GCN(3, 2, 3)
    print(model(x, adj).shape)