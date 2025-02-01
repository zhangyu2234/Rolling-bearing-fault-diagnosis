import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_dir = './student.csv'
df = pd.read_csv(data_dir, delimiter=',')
df = df.drop(columns='Course').reset_index(drop=True)
# Df -> numpy
data = df.to_numpy(dtype=np.float32)
# data -> 4, 5 -> 5个样本，每个样本4个特征
x_mean = np.mean(data, axis=1)

# correlation matrix
def corr_mx(x):
    corr_mx = np.zeros((x.shape[0], x.shape[0]))
    n = x.shape[1]
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            corr = np.sum(((x[i, :] - x_mean[i]) * (x[j, :] - x_mean[j]))) / (n-1)
            std_ii = np.sqrt((np.sum((x[i, :] - x_mean[i])**2) / (n-1)))
            std_jj = np.sqrt((np.sum((x[j, :] - x_mean[j])**2) / (n-1)))

            corr_mx[i, j] = corr / (std_ii*std_jj)
    return corr_mx

R = corr_mx(data) # 成绩的相关性矩阵
eign_val, eign_vector = np.linalg.eig(R)

# 将主成分方差（协方差矩阵特征值）由大到小进行排列
eign_val_sort = np.sort(eign_val)[::-1]

# 假设要求主成分的累计方差贡献率大于75%
val_sum = np.sum(eign_val_sort)
print((eign_val_sort[0] + eign_val_sort[1]) / val_sum) # 只需选取前两个特征值即可

lamda1, lamda2 = eign_val_sort[0], eign_val_sort[1]
vec1, vec2 = eign_vector[-1], eign_vector[-2]
vec = np.vstack((vec1, vec2))


data_mean = np.mean(data, axis=1)
data = data - data_mean[:, np.newaxis]
pca = vec @ data
print(pca.shape)
pca = pca.T # 样本数，特征数 5,2

# Vis
plt.figure(figsize=(8, 5))
colors = np.linspace(0, 1, len(pca[:, 0]))
plt.scatter(pca[:, 0], pca[:, 1], c=colors, cmap='rainbow', edgecolors='k', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()










