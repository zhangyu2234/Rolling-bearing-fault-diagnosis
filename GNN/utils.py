import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

# F-范数

def Frobenius(mx):
    assert mx.shape[0] == mx.shape[1]
    mx = mx * mx
    mx = np.sum(mx)
    return np.sqrt(mx)


def EU_dist(mx):
    aa = np.sum((mx*mx), axis=1)
    ab = np.dot(mx, mx.T)
    dis_mat = aa + aa.T - 2*ab
    dis_mat[dis_mat < 0] = 0
    dis_mat = np.sqrt(dis_mat)
    return np.maximum(dis_mat, dis_mat.T)
    

def construct_A_with_KNN_from_distance(dis_mat, k=None):
    N = dis_mat.shape[0]
    A = np.zeros((N, N))

    for center_idx in range(N):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        nearest_idx = nearest_idx[:k]

        if not np.any(center_idx == nearest_idx):
            nearest_idx[-1] = center_idx
        
        for node_idx in nearest_idx:
            A[node_idx, center_idx] = 1.0
    
    return A


def Normalize(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0
    d_mat_inv = sp.diags(d_inv)
    mx = d_mat_inv.dot(adj)
    return mx

def sparse_to_tensor(mx):
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack([mx.row, mx.col]).astype(np.int64)
    )
    values = torch.from_numpy(mx.data)

    shape = torch.Size(mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def split_dataset(label):
    train_prop = .5
    valid_prop = .25

    indices = []
    for i in range(label.max()+1):
        index = torch.where(label == i)[0].view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    percls_trn = int(train_prop / (label.max()+1) * len(label))
    val_lb = int(valid_prop * len(label))

    train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
    reset_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    reset_index = reset_index[torch.randperm(reset_index.size(0))]

    valid_index = reset_index[:val_lb]

    test_index = reset_index[val_lb:]

    return train_idx, valid_index, test_index


def valid(pred, rel):

    idx = torch.argmax(pred, dim=1)
    total_pred = (idx == rel).sum().item()

    return total_pred / len(rel)


def train_GCN(args, model, x, adj, label):
    epochs = args.epochs
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_idx, valid_idx, test_idx = split_dataset(label)
    best_val_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        model.train()
        pred = model(x, adj)
        train_loss = F.nll_loss(pred[train_idx], label[train_idx])
        train_acc = valid(pred[train_idx], label[train_idx])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, adj)
            val_loss = F.nll_loss(pred[valid_idx], label[valid_idx])
            val_acc = valid(pred[valid_idx], label[valid_idx])

        
        if (epoch % 10) == 0:
            print('train_loss: {:.4f}, train_acc: {:.2f}, val_loss: {:.4f}, val_acc: {:.2f}'.format(train_loss, train_acc, val_loss, val_acc))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), './GNN/best_model.pth')

        
        else:
            patience += 1
        
        if patience >= 50:
            break
    
    model.load_state_dict(torch.load('./GNN/best_model.pth'))
    model.eval()
    with torch.no_grad():
        pred = model(x, adj)
    
    acc = accuracy_score(label[test_idx].detach().cpu().numpy(), pred[test_idx].detach().cpu().numpy().argmax(axis=-1))
    f1= f1_score(label[test_idx].detach().cpu().numpy(), pred[test_idx].detach().cpu().numpy().argmax(axis=-1), average='micro')
    
    print(f'ACC: {acc:.2f}, F1: {f1:.2f}')



    

if __name__ =="__main__":
    a = np.random.randn(5, 5)
    dis = EU_dist(a)
    A = construct_A_with_KNN_from_distance(dis, k=2)

    mx = Normalize(A)
    print(mx)
    
