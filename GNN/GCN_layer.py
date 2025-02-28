import torch
import torch.nn as nn

class GCN_layer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=None):
        super(GCN_layer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias

        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim))

        if self.bias is not None:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    
    def forward(self, x, adj):
        x = torch.mm(x, self.W)
        
        output =  adj @ x

        if self.bias is not None:
            output = output + self.bias

        return output
    


if __name__ =='__main__':
    x = torch.randn(5, 3)
    adj = torch.randn(5, 5)

    layer = GCN_layer(3, 3)
    print(layer(x, adj).shape)
        
        

        

    
    
      

    
