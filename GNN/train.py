import pickle
from GNN.GCN import GCN
from GNN.utils import *
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='dataset/JNU/dataset.cpkl', help='load data')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=float, default=200, help='epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--hidden_channel', type=int, default=256, help='hidden')
parser.add_argument('--model', type=str, default='GCN', help='model')

args = parser.parse_args()

with open(args.data_path, 'rb') as f:
    data, label = pickle.load(f)
f.close()

print(type(data))
print(type(label))

# adj
dis_mx = EU_dist(data)
adj = construct_A_with_KNN_from_distance(dis_mat=dis_mx, k=3)
print(type(adj))

# numpy -> tensor
data = torch.FloatTensor(data).cuda()
label = torch.LongTensor(label).cuda()
adj = torch.FloatTensor(adj)

# Normalize 
adj = sparse_to_tensor(Normalize(adj)).cuda()

model = GCN(data.shape[1], args.hidden_channel, label.max().item() + 1).cuda()


if __name__ == '__main__':
    train_GCN(args, model, data, adj, label)

    













