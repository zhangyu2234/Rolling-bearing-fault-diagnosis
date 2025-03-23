import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

num_nodes = 20
adj = np.zeros((num_nodes, num_nodes), dtype=int)

# 构造 ring graph 的邻接矩阵
for i in range(num_nodes): # i=3
    adj[i, (i + 1) % num_nodes] = 1  # 3, 0
    adj[i, (i - 1) % num_nodes] = 1 # 3, 2 % 4 = 2 -> 3, 2

# i=0
# 0, 0 % 4=1 -> 0, 1
# 0, -1 % 4=3 -> 0, 3

# i=1
# 1, 2%4 -> 1, 2
# 1, 0%4 -> 0 

degrees = np.sum(adj, axis=1)

# 构造度矩阵（对角矩阵）
D = np.diag(degrees)
L = D - adj

eignval, eignvec = np.linalg.eigh(L)

""" 
plt.figure(figsize=(10, 5))
for i in range(len(eignval)):
    plt.subplot(5, 4, i+1)
    plt.plot(eignvec[:, i])
    plt.title('eignvec = {}'.format(eignval[i]))
plt.tight_layout()
plt.show()
"""

# vis graph
plt.figure(figsize=(10, 5))
G = nx.from_numpy_array(adj)
nx.draw(G)
plt.show()