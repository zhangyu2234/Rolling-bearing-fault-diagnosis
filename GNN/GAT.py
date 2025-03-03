import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT_layer import GraphAttentionLayer


class GraphAttentionModel(nn.Module):
    def __init__(self, in_channel, out_channel, n_class, num_heads, dropout):
        super(GraphAttentionModel, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_class = n_class

        self.layer = nn.ModuleList()

        self.layer.extend([GraphAttentionLayer(in_channel, out_channel, dropout=dropout) for _ in range(num_heads)])

        self.out_layer = GraphAttentionLayer(num_heads*out_channel, n_class, dropout=dropout)
    

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = []
        for i in range(self.num_heads):
            x1 = self.layer[i](x, adj)
            out.append(x1)
        
        x = torch.cat(out, dim=1)
        out = self.out_layer(x, adj)

        return F.log_softmax(out, dim=-1)


if __name__ =='__main__':
    x = torch.randn(2708, 1433)
    adj = torch.randn(2708, 2708)

    model = GraphAttentionModel(1433, 256, 7, 2, dropout=0.2)
    print(model(x, adj).shape)



   