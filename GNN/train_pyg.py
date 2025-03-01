import pickle
from GNN.GCN_pyg import GCN_Pyg
from GNN.utils import *
import argparse
import scipy.sparse as sp
from torch_geometric.data import Data


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='dataset/JNU/dataset.cpkl', help='load data')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=float, default=200, help='epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--hidden_channel', type=int, default=256, help='hidden')
parser.add_argument('--model', type=str, default='GCN', help='model')

args = parser.parse_args()

with open(args.data_path, 'rb') as f:
    fts, label = pickle.load(f)
f.close()


# adj
dis_mx = EU_dist(fts)
adj = construct_A_with_KNN_from_distance(dis_mat=dis_mx, k=3)
print(type(adj))

# numpy -> tensor
x = torch.FloatTensor(fts).cuda()
label = torch.LongTensor(label).cuda()
adj = torch.FloatTensor(adj)

# Pyg data
adj = sparse_to_tensor(Normalize(adj)).cuda()

edge_index = adj.coalesce().indices()
edge_weights = adj.coalesce().values()
print(edge_weights)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights)

model = GCN_Pyg(data.x.shape[1], args.hidden_channel, label.max().item() + 1, args.dropout).cuda()

if __name__ =='__main__':
    train_GCNConv(args, model, data.x, data.edge_index, data.edge_attr, label)

    















    













