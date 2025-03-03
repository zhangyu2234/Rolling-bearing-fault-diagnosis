import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.2):
        super(GraphAttentionLayer, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dropout = dropout

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

        self.W = nn.Parameter(torch.FloatTensor(in_channel, out_channel))

        self.a = nn.Parameter(torch.FloatTensor(2 * out_channel, 1))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    

    def forward(self, x, adj):
        Wh = x @ self.W # N, out_channel
        Wh1 = torch.matmul(Wh, self.a[:self.out_channel, :]) # N, 1
        Wh2 = torch.matmul(Wh, self.a[self.out_channel:, :]) # N, 1
        e = self.leakyrelu((Wh1 + Wh2.T)) # N, N
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)
    

if __name__ =='__main__':
    x = torch.randn(2708, 1433)
    adj = torch.randn(2708, 2708)

    layer = GraphAttentionLayer(1433, 256)
    print(layer(x, adj).shape)


        

    





